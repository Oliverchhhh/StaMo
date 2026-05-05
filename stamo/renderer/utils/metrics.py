from datetime import timedelta
from timeit import default_timer

import torch
import torch.nn.functional as F


def calculate_psnr(pred, target, max_val=1.0):
    """
    计算批量图像的 PSNR
    pred, target: [B, C, H, W] 取值范围 [0, max_val]
    """
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])  # 每张图像的 MSE
    psnr = 10 * torch.log10((max_val**2) / mse)
    return psnr.mean()


def calculate_ssim(pred, target, max_val=1.0, window_size=11, K1=0.01, K2=0.03):
    """
    计算批量图像的 SSIM（彩色）
    pred, target: [B, C, H, W] 范围 [0, max_val]
    """
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    device = pred.device

    def gaussian_window(window_size, sigma):
        gauss = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
        gauss = torch.exp(-(gauss**2) / (2 * sigma**2))
        return gauss / gauss.sum()

    sigma = 1.5
    gauss_1d = gaussian_window(window_size, sigma).unsqueeze(1)
    window_2d = gauss_1d @ gauss_1d.T
    window = window_2d.expand(pred.size(1), 1, window_size, window_size).to(device=device)  # [C, 1, H, W]

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=pred.size(1))
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=target.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=pred.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=target.size(1)) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=pred.size(1)) - mu1_mu2

    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def get_parameters(net: torch.nn.Module):
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    fp32_trainable_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.float32 and p.requires_grad)
    fp16_trainable_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.float16 and p.requires_grad)
    bf16_trainable_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.bfloat16 and p.requires_grad)
    fp32_frozen_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.float32 and not p.requires_grad)
    fp16_frozen_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.float16 and not p.requires_grad)
    bf16_frozen_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.bfloat16 and not p.requires_grad)
    return {
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_fp32": fp32_trainable_params,
        "trainalbe_fp16": fp16_trainable_params,
        "trainalbe_bf16": bf16_trainable_params,
        "frozen_fp32": fp32_frozen_params,
        "frozen_fp16": fp16_frozen_params,
        "frozen_bf16": bf16_frozen_params,
    }


class Meter:
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n: int = 1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        if isinstance(val, (int, float)):
            self._update_scalar(val, n)
        elif isinstance(val, dict):
            self._update_dict(val, n)
        else:
            raise ValueError(f"Not supported type {type(val)}")

    def _update_scalar(self, val: float, n: int):
        self.val = val
        self.sum = self.sum + val * n if self.sum is not None else val * n
        self.count = self.count + n if self.count is not None else n
        self.avg = self.sum / self.count

    def _update_dict(self, val: dict, n: int):
        # tensor -> item
        val = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in val.items()}

        self.val = {**self.val, **val} if self.val is not None else val

        if self.sum is not None:
            for k, v in val.items():
                self.sum[k] = self.sum.get(k, 0) + v * n
        else:
            self.sum = {k: v * n for k, v in val.items()}

        if self.count is not None:
            for k in val.keys():
                self.count[k] = self.count.get(k, 0) + n
        else:
            self.count = dict.fromkeys(val.keys(), n)

        self.avg = {k: self.sum[k] / self.count[k] for k in self.count.keys()}

    def __str__(self):
        if isinstance(self.avg, dict):
            return str({k: f"{v:.4f}" for k, v in self.avg.items()})
        return "Nan"


class Timer:
    def __init__(self):
        """
        t = Timer()
        time.sleep(1)
        print(t.elapse())
        """
        self.start = default_timer()

    def elapse(self, readable=False):
        seconds = default_timer() - self.start
        if readable:
            seconds = str(timedelta(seconds=seconds))
        return seconds
