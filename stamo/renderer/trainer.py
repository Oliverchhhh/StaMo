import os
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as DIST
import torchvision.transforms as T
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from stamo.renderer.model.renderer import RenderNet
from stamo.renderer.utils.data import complex_to_device, fp32_to_bf16, fp32_to_fp16, move_to_cuda
from stamo.renderer.utils.files import ensure_directory, ensure_dirname
from stamo.renderer.utils.metrics import Meter, Timer, calculate_psnr, calculate_ssim, get_parameters
from stamo.renderer.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


class Trainer:
    def __init__(self, args, model: RenderNet, criterion=None, optimizer=None, lr_scheduler=None) -> None:
        self.model: RenderNet = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.local_rank = overwatch.local_rank()
        self.rank = overwatch.rank()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.epoch = -1
        self.global_step = -1

        self.eval_before_train = False

        self.use_deepspeed = args.deepspeed
        self.use_fabric = args.fabric

        self.resume = args.resume
        self.resume_path = args.resume_path
        self.do_train = args.do_train

        self.num_iters = args.train.num_iters
        self.epochs = args.train.epochs
        self.eval_step = args.train.eval_step
        self.save_step = args.train.save_step
        self.local_batch_size = args.train.local_batch_size
        self.gradient_accumulate_steps = args.train.gradient_accumulate_steps
        self.iter_per_ep = None

        self.seed = args.seed
        self.task_name = args.task_name
        self.img_size = args.data.img_size
        self.log_dir = os.path.join(args.log_dir, args.task_name)
        self.ckpt_save_dir = os.path.join(args.train.ckpt_save_dir, args.task_name)

        if overwatch.is_rank_zero() and args.do_train:
            ensure_directory(self.log_dir)
            self.writer = SummaryWriter(
                log_dir=self.log_dir, comment="StaMo Renderer"
            )  # 这里的logs要与--logdir的参数一样

    def move_model_to_cuda(self) -> None:
        self.model.to(self.device)
        if self.optimizer is not None:
            if isinstance(self.optimizer, list):
                for i in range(len(self.optimizer)):
                    self.optimizer[i].load_state_dict(
                        complex_to_device(self.optimizer[i].state_dict(), device=self.device)
                    )
            else:
                self.optimizer.load_state_dict(complex_to_device(self.optimizer.state_dict(), device=self.device))

    def prepare_dist_model(self) -> None:
        tb = TensorBoardLogger(root_dir=self.log_dir, version=0)
        self.fabric = Fabric(loggers=tb)
        if hasattr(self.fabric.strategy, "config") and isinstance(self.fabric.strategy.config, dict):
            self.fabric.strategy.config["gradient_clipping"] = 1.0
        if self.resume:
            assert os.path.exists(self.resume_path)

            overwatch.warning(f"Resuming from {self.resume_path}")
            self.load_checkpoint(self.resume_path)

        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

        if not self.do_train:
            self.model.eval()
        overwatch.info(f"Successfully built models with {get_parameters(self.model)} parameters")

    def forward_step(self, inputs, **kwargs) -> Dict[str, Any]:
        outputs = self.model(inputs, **kwargs)
        return outputs

    def backward_step(self, loss) -> None:
        if self.use_deepspeed:
            self.model.backward(loss)
        elif self.use_fabric:
            self.fabric.backward(loss)
        else:
            loss.backward()

    def prepare_batch(self, batch) -> Dict[str, Any]:
        batch = move_to_cuda(batch)
        batch = fp32_to_bf16(batch)
        if self.use_deepspeed:
            batch = fp32_to_fp16(batch)
        return batch

    def step(self, optimizer_idx=-1) -> None:
        if optimizer_idx >= 0 and isinstance(self.optimizer, list):
            optimizer = self.optimizer[optimizer_idx]
        else:
            optimizer = self.optimizer
        optimizer.step()
        optimizer.zero_grad()

    def reduce_mean(self, v) -> float:
        world_size = overwatch.world_size()
        if world_size < 2:
            return v
        else:
            t = v.clone().detach().cuda()
            DIST.all_reduce(t)
            t = t.item() / world_size
        return t

    def save_checkpoint(self) -> None:
        save_path = os.path.join(self.ckpt_save_dir, str(self.global_step))
        overwatch.warning(f"Saving models to {save_path}")
        if self.use_deepspeed:
            self.model.save_checkpoint(save_path, self.global_step)
        else:
            if overwatch.is_rank_zero():
                ensure_directory(save_path)
                self.model.save_checkpoint(save_path, self.global_step)

    def load_checkpoint(self, load_path) -> None:
        global_step = self.model.load_checkpoint(load_path)
        self.global_step = global_step

    def setup_model_for_training(self) -> None:
        if overwatch.is_rank_zero():
            overwatch.warning(f"Existing dirs detected {self.log_dir}")
            ensure_dirname(self.log_dir, override=False)

        self.model.set_trainable_params()
        if not self.use_fabric:
            self.move_model_to_cuda()
        self.prepare_dist_model()

    def train_eval_by_iter(self, train_loader, eval_loader=None, use_tqdm=True) -> None:
        if self.use_fabric:
            train_loader = self.fabric.setup_dataloaders(train_loader, use_distributed_sampler=False)
        if self.num_iters:
            overwatch.warning("Start train & val phase...")
        else:
            overwatch.warning("Skip train & val phase...")
            return
        overwatch.warning(
            f"Train examples: {len(train_loader.dataset)},\n"
            f"Val examples: {len(eval_loader.dataset)}, {len(eval_loader)}\n"
            f"epochs: {self.epochs}, iters: {self.num_iters}, \n"
            f"eval_step: {self.eval_step}, save_step: {self.save_step},\n"
            f"global_batch_size: {self.local_batch_size * overwatch.world_size() * self.gradient_accumulate_steps}, local_batch_size: {self.local_batch_size}."
        )

        # Train & Eval phase
        train_pbar = tqdm(total=self.num_iters, disable=not use_tqdm)
        train_meter = Meter()

        if self.global_step > 0:
            train_pbar.update(self.global_step)
        else:
            self.global_step = 0

        if self.eval_before_train and self.global_step == 0:
            if eval_loader:
                eval_meter, eval_time = self.eval_fn(eval_loader, use_tqdm=use_tqdm)
                overwatch.info(f"[Rank {self.rank}] Valid before train. Time: {eval_time}\n{eval_meter.avg}")

        self.model.train()

        while True:
            train_iter = iter(train_loader)
            while self.global_step < self.num_iters:
                try:
                    inputs = next(train_iter)
                except StopIteration:
                    # overwatch.warning("Reaching end of the train_loader, terminating training loop")
                    break

                self.epoch = (self.global_step + 1) // self.iter_per_ep
                if not getattr(self.optimizer, "is_enabled", lambda x: True)(self.global_step):
                    continue  # adjust to sdm KL-VAE

                inputs["epoch"] = self.epoch
                inputs["global_step"] = self.global_step
                is_accumulating = self.global_step % self.gradient_accumulate_steps != 0

                with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                    inputs = self.prepare_batch(inputs)
                    outputs = self.forward_step(inputs, criterion=self.criterion)
                    self.backward_step(outputs["loss"])

                loss_to_log = outputs["loss"].item()
                self.fabric.log("loss", loss_to_log, step=self.global_step)
                if not is_accumulating:
                    self.step()
                    self.lr_scheduler.step()

                metric_and_loss = {k: v for k, v in outputs.items() if k.split("_")[0] in ["metric", "loss"]}
                for k, v in metric_and_loss.items():
                    metric_and_loss[k] = self.reduce_mean(v)
                train_meter.update(metric_and_loss)
                train_pbar.set_description("Metering: " + str(train_meter))

                if self.global_step % self.save_step == 0 and self.global_step != 0:
                    overwatch.warning("Saving model...")
                    self.save_checkpoint()

                if self.global_step % self.eval_step == 0 and self.global_step != 0:
                    overwatch.warning("Evaluating...")
                    if eval_loader:
                        # sample a single round training sample to test whether over-fitting
                        eval_meter, eval_time = self.eval_fn(eval_loader, use_tqdm=use_tqdm)
                        overwatch.info(
                            f"[Rank {self.rank}] Valid Step: {self.global_step}, Time: {eval_time}\n{eval_meter.avg}"
                        )

                        # Update metric with eval metrics
                    train_meter = Meter()

                if overwatch.is_rank_zero():
                    for k, v in metric_and_loss.items():
                        self.writer.add_scalar(k, v, self.global_step)

                self.global_step += 1
                train_pbar.update(1)

            if self.global_step >= self.num_iters:
                break

        if self.global_step % self.save_step != 0:
            overwatch.warning("Saving model...")
            self.save_checkpoint()

        if self.global_step % self.eval_step != 0:
            overwatch.warning("Evaluating...")
            if eval_loader:
                eval_meter, eval_time = self.eval_fn(eval_loader, use_tqdm=use_tqdm)
                overwatch.info(
                    f"[Rank {self.rank}] Valid Step: {self.global_step}, Time: {eval_time}\n{eval_meter.avg}"
                )

    def eval_fn(self, eval_loader, use_tqdm=True):
        # TODO Note that eval_fn supports ddp. So we do not need to unwrap things here.
        self.model.eval()
        eval_meter = Meter()
        eval_timer = Timer()

        label_imgs = []
        pred_imgs = []

        with torch.no_grad():
            eval_loader = tqdm(eval_loader, total=len(eval_loader)) if use_tqdm else eval_loader
            for inputs in eval_loader:
                inputs = self.prepare_batch(inputs)
                outputs = self.forward_step(inputs)
                metric_and_loss = {k: v for k, v in outputs.items() if k.split("_")[0] in ["metric", "loss"]}

                for k, v in metric_and_loss.items():
                    metric_and_loss[k] = self.reduce_mean(v)
                eval_meter.update(metric_and_loss)

                label_img = inputs["images"]

                pred_img = self.model.inv_vae_transform(outputs["images"])
                pred_img = torch.clamp(pred_img, 0, 1)

                label_imgs.append(label_img)
                pred_imgs.append(pred_img)

            label_imgs = torch.cat(label_imgs, dim=0)
            pred_imgs = torch.cat(pred_imgs, dim=0)

            overwatch.info(f"PSNR: {calculate_psnr(pred_imgs, label_imgs):.4f}")
            overwatch.info(f"SSIM: {calculate_ssim(pred_imgs, label_imgs):.4f}")

            np_images = np.stack([np.asarray(img.permute(0, 2, 1).cpu().float()) for img in pred_imgs])
            np_gt_images = np.stack([np.asarray(img.permute(0, 2, 1).cpu().float()) for img in label_imgs])

            toimg = T.ToPILImage()

            images = [toimg(img.cpu().float()) for img in pred_imgs]
            gt_images = [toimg(img.cpu().float()) for img in label_imgs]

            if overwatch.is_rank_zero():
                image_path = os.path.join(self.log_dir, "images", str(self.global_step))
                ensure_directory(os.path.join(image_path))
                for i in range(len(images)):
                    images[i].save(os.path.join(image_path, f"{i}_pred.jpeg"))
                    gt_images[i].save(os.path.join(image_path, f"{i}_gt.jpeg"))

                self.writer.add_images("validation/pred", np_images, self.global_step, dataformats="NCWH")
                self.writer.add_images("validation/gt", np_gt_images, self.global_step, dataformats="NCWH")

        eval_time = eval_timer.elapse(True)

        self.model.train()
        return eval_meter, eval_time

    def manually_eval(self, images, batch_size=64):
        self.model.eval()

        label_imgs = images
        toimg = T.ToPILImage()
        transforms = T.Compose(
            [T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()]
        )

        image_path = os.path.join(self.log_dir, "images", str(self.global_step))
        ensure_directory(os.path.join(image_path))

        with torch.no_grad():
            for start_idx in range(0, len(images), batch_size):
                end_idx = min(start_idx + batch_size, len(images))
                batch_images = images[start_idx:end_idx]

                tensor_images = torch.stack([transforms(image).to(self.device) for image in batch_images])
                inputs = {"images": tensor_images}

                inputs = self.prepare_batch(inputs)
                outputs = self.forward_step(inputs)

                pred_imgs = self.model.inv_vae_transform(outputs["images"])
                pred_imgs = torch.clamp(pred_imgs, 0, 1)

                overwatch.info(f"PSNR: {calculate_psnr(pred_imgs, tensor_images):.4f}")
                overwatch.info(f"SSIM: {calculate_ssim(pred_imgs, tensor_images):.4f}")

                pred_imgs = [toimg(pred_img.squeeze().cpu()) for pred_img in pred_imgs]

                for idx, pred_img in enumerate(pred_imgs):
                    pred_img.save(os.path.join(image_path, f"{start_idx + idx}_pred.jpeg"))
                    label_imgs[idx].save(os.path.join(image_path, f"{start_idx + idx}_gt.jpeg"))

    def interpolation_eval(
        self,
        image1,
        image2,
        tokens=None,
        num_interpolation=5,
        to_video=False,
        name="interpolation.mp4",
    ):
        """
        对压缩token进行线性插值
        """
        self.model.eval()

        transforms = T.Compose(
            [T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()]
        )

        with torch.no_grad():
            image1 = transforms(image1).to(self.device).unsqueeze(0)
            image2 = transforms(image2).to(self.device).unsqueeze(0)

            inputs1 = self.prepare_batch(image1)
            inputs2 = self.prepare_batch(image2)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)

            outputs = self.model.interpolation_eval(
                inputs1,
                inputs2,
                generator,
                tokens=tokens,
                num_interpolation=num_interpolation,
            )

        toimg = T.ToPILImage()

        images = []
        for pred_image in outputs:
            pred_image = self.model.inv_vae_transform(pred_image)
            pred_image = torch.clamp(pred_image, 0, 1)
            images.append(toimg(pred_image.cpu()))

        if to_video:
            import imageio

            video_path = os.path.join(self.log_dir, "images", str(self.global_step))
            ensure_directory(video_path)
            save_path = os.path.join(video_path, name)
            imageio.mimsave(save_path, images, fps=10)
            return

        image_path = os.path.join(self.log_dir, "images", str(self.global_step))
        ensure_directory(image_path)
        for i in range(len(images)):
            images[i].save(os.path.join(image_path, f"interpolation_{i}.jpeg"))

        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths)
        max_height = max(heights)

        combined_image = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.size[0]

        # 保存拼接后的图像
        combined_image.save(os.path.join(image_path, f"combined_step_{self.global_step}.jpeg"))

    def delta_interpolation(self, image, start, end):
        """
        进行delta插值
        """
        self.model.eval()

        toimg = T.ToPILImage()
        transforms = T.Compose(
            [T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()]
        )
        size = image.size

        with torch.no_grad():
            start_inputs = transforms(start).to(self.device).unsqueeze(0)
            end_inputs = transforms(end).to(self.device).unsqueeze(0)
            image_inputs = transforms(image).to(self.device).unsqueeze(0)

            start_inputs = self.prepare_batch(start_inputs)
            end_inputs = self.prepare_batch(end_inputs)
            image_inputs = self.prepare_batch(image_inputs)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)

            outputs = self.model.delta_interpolation(
                image_inputs,
                start_inputs,
                end_inputs,
                generator,
            )

        pred_image = self.model.inv_vae_transform(outputs).squeeze(0)
        pred_image = torch.clamp(pred_image, 0, 1)
        pred_image = toimg(pred_image.cpu())

        image_path = os.path.join(self.log_dir, "images", str(self.global_step))
        ensure_directory(os.path.join(image_path))

        pred_image.save(os.path.join(image_path, f"delta_interpolation_{self.global_step}.jpeg"))

        images = [start.resize(size), end.resize(size), image, pred_image.resize(size)]
        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths)
        max_height = max(heights)

        combined_image = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.size[0]

        combined_image.save(os.path.join(image_path, f"delta_interpolation_combined_{self.global_step}.jpeg"))
