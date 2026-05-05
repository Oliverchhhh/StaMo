import os

from PIL import Image

from stamo.renderer.model.renderer import RenderNet
from stamo.renderer.trainer import Trainer
from stamo.renderer.utils.args import init_args
from stamo.renderer.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


def main_worker(args):
    args.train.iter_per_ep = None
    overwatch.info("Building models...")
    model = RenderNet(args)

    trainer = Trainer(args=args, model=model)
    trainer.load_checkpoint(args.resume_path)
    trainer.move_model_to_cuda()

    # 得到重建结果以及PSNR和SSIM
    root_dir = "test/libero/episode0"
    image_dir = os.listdir(root_dir)

    image_ids = [f"{root_dir}/{img}" for img in image_dir]
    images = [Image.open(image_id).convert("RGB") for image_id in image_ids]

    trainer.manually_eval(images, batch_size=512)

    # 得到首尾两帧之间的线性插值结果
    image1 = Image.open("test/libero/episode0/libero_000352_0000.jpg").convert("RGB")
    image2 = Image.open("test/libero/episode0/libero_000352_0040.jpg").convert("RGB")

    trainer.interpolation_eval(image1, image2, tokens=[], num_interpolation=60, to_video=True, name="interpolation.mp4")
    # trainer.interpolation_eval(image1, image2, tokens=[1], num_interpolation=60, to_video=True, name="interpolation_0.mp4")
    # trainer.interpolation_eval(image1, image2, tokens=[0], num_interpolation=60, to_video=True, name="interpolation_1.mp4")

    # 得到action迁移的结果 image + end - start
    image = Image.open("test/libero/episode2/libero_000376_0000.jpg").convert("RGB")
    start = Image.open("test/libero/episode0/libero_000352_0000.jpg").convert("RGB")
    end = Image.open("test/libero/episode0/libero_000352_0040.jpg").convert("RGB")

    trainer.delta_interpolation(image, end, start)


if __name__ == "__main__":
    config = init_args()
    main_worker(config)
