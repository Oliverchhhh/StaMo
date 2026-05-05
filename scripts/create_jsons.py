import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List

from stamo.renderer.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)

random.seed(33)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def find_leaf_dirs(root: str) -> List[str]:
    """返回所有包含图片文件的叶子目录（chunk级别目录）"""
    leaf_dirs = []
    for dirpath, subdirs, files in os.walk(root):
        has_images = any(Path(f).suffix.lower() in IMAGE_EXTENSIONS for f in files)
        if has_images:
            leaf_dirs.append(dirpath)
    return sorted(leaf_dirs)


def create_split_jsonl(train_root: str, eval_root: str, dataset_name: str, eval_samples_per_chunk: int = 3):
    train_leaf_dirs = find_leaf_dirs(train_root)
    eval_leaf_dirs = find_leaf_dirs(eval_root)

    overwatch.info(f"发现 {len(train_leaf_dirs)} 个 train chunk 目录")
    overwatch.info(f"发现 {len(eval_leaf_dirs)} 个 eval chunk 目录")

    os.makedirs("./jsons", exist_ok=True)

    eval_images_entries = []
    cnt = 0

    # 每个 train chunk 目录单独写一个 jsonl，并各自抽取若干张进 eval
    for chunk_dir in train_leaf_dirs:
        images = sorted([
            os.path.abspath(os.path.join(chunk_dir, f))
            for f in os.listdir(chunk_dir)
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        ])

        if len(images) < eval_samples_per_chunk:
            overwatch.warning(f"跳过图片不足的目录: {chunk_dir} ({len(images)} 张)")
            continue

        suffix = f"part_{cnt}"
        train_jsonl_path = f"./jsons/train_{dataset_name}_{suffix}.jsonl"
        cnt += 1

        shared_for_eval = random.sample(images, eval_samples_per_chunk)
        for img in shared_for_eval:
            eval_images_entries.append({"image": img, "//": "from-train-shared"})

        with open(train_jsonl_path, "w", encoding="utf-8") as f:
            for img in images:
                f.write(json.dumps({"image": img}) + "\n")

    overwatch.info(f"Train: 共 {cnt} 个 chunk，每个抽 {eval_samples_per_chunk} 张进 eval")

    # 每个 eval chunk 目录各自抽取若干张进 eval
    for chunk_dir in eval_leaf_dirs:
        images = sorted([
            os.path.abspath(os.path.join(chunk_dir, f))
            for f in os.listdir(chunk_dir)
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        ])

        if len(images) < eval_samples_per_chunk:
            overwatch.warning(f"跳过图片不足的目录: {chunk_dir} ({len(images)} 张)")
            continue

        eval_only = random.sample(images, eval_samples_per_chunk)
        for img in eval_only:
            eval_images_entries.append({"image": img, "//": "from-eval-only"})

    eval_jsonl_path = f"./jsons/eval_{dataset_name}.jsonl"
    with open(eval_jsonl_path, "w", encoding="utf-8") as f:
        for entry in eval_images_entries:
            f.write(json.dumps(entry) + "\n")

    overwatch.info(f"Eval: 共 {len(eval_images_entries)} 张，写入 {eval_jsonl_path}")

    # 写 train json（多 part 配置）
    datasets = [f"train_{dataset_name}_part_{i}.jsonl" for i in range(cnt)]
    ratios = [1 / cnt for _ in range(cnt)]
    train_json_path = f"./jsons/train_{dataset_name}.json"
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump({"datasets": datasets, "ratios": ratios}, f, indent=4)
    overwatch.info(f"Train config 写入 {train_json_path}，共 {cnt} parts")

    # 写 eval json
    eval_json_path = f"./jsons/eval_{dataset_name}.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump({"datasets": [f"eval_{dataset_name}.jsonl"], "ratios": [1]}, f, indent=4)
    overwatch.info(f"Eval config 写入 {eval_json_path}")


if __name__ == "__main__":
    create_split_jsonl(
        train_root="/mnt/nas/datasets4/open-p2p/datasets/train",
        eval_root="/mnt/nas/datasets4/open-p2p/datasets/eval",
        dataset_name="cuphead",
        eval_samples_per_chunk=3,
    )
