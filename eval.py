import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# =========================
# 配置你的图片目录
# =========================
image_dir = "/home/ch/StaMo/logs/cuphead336/images/26600"

# 支持的图片后缀
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

psnr_list = []
ssim_list = []

# 找到所有 *_gt.xxx
gt_files = []

for fname in os.listdir(image_dir):
    lower = fname.lower()
    if any(lower.endswith(ext) for ext in IMG_EXTS) and "_gt" in lower:
        gt_files.append(fname)

gt_files.sort()

print(f"找到 {len(gt_files)} 组图片")

for gt_name in gt_files:

    # 构造 pred 文件名
    pred_name = gt_name.replace("_gt", "_pred")

    gt_path = os.path.join(image_dir, gt_name)
    pred_path = os.path.join(image_dir, pred_name)

    if not os.path.exists(pred_path):
        print(f"缺少预测图片: {pred_name}")
        continue

    # 读取图片
    gt_img = cv2.imread(gt_path)
    pred_img = cv2.imread(pred_path)

    if gt_img is None or pred_img is None:
        print(f"读取失败: {gt_name} / {pred_name}")
        continue

    # 尺寸不一致时 resize
    if gt_img.shape != pred_img.shape:
        pred_img = cv2.resize(
            pred_img,
            (gt_img.shape[1], gt_img.shape[0])
        )

    # BGR -> RGB
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

    # =========================
    # 计算 PSNR
    # =========================
    psnr = peak_signal_noise_ratio(
        gt_img,
        pred_img,
        data_range=255
    )

    # =========================
    # 计算 SSIM
    # =========================
    ssim = structural_similarity(
        gt_img,
        pred_img,
        channel_axis=2,
        data_range=255
    )

    psnr_list.append(psnr)
    ssim_list.append(ssim)

    print(
        f"{gt_name} vs {pred_name} | "
        f"PSNR: {psnr:.4f}, "
        f"SSIM: {ssim:.4f}"
    )

# =========================
# 输出平均指标
# =========================
if len(psnr_list) > 0:
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print("\n====================")
    print(f"平均 PSNR: {avg_psnr:.4f}")
    print(f"平均 SSIM: {avg_ssim:.4f}")
    print("====================")
else:
    print("没有成功计算任何图片对")