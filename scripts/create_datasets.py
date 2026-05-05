import cv2
import os
import random
import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count


BASE_DIR = Path("/mnt/nas/datasets4/open-p2p/datasets").absolute()
TRAIN_ROOT = BASE_DIR / "train"
EVAL_ROOT = BASE_DIR / "eval"


def process_one_video(args):
    video_path, rel_path, video_name, fps, split_ratio = args

    curr_train_dir = TRAIN_ROOT / rel_path / video_name
    curr_eval_dir = EVAL_ROOT / rel_path / video_name

    train_done = curr_train_dir.exists() and any(curr_train_dir.iterdir())
    eval_done = curr_eval_dir.exists() and any(curr_eval_dir.iterdir())
    if train_done and eval_done:
        print(f"  [跳过] 已存在: {video_name}")
        return "skipped", video_path, 0, 0

    curr_train_dir.mkdir(parents=True, exist_ok=True)
    curr_eval_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [错误] 无法打开视频: {video_path}")
        return "failed", video_path, 0, 0

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30
    frame_interval = max(1, int(round(original_fps / fps)))

    frame_idx = 0
    t_count = 0
    e_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            is_train = random.random() < split_ratio
            if is_train:
                save_path = curr_train_dir
                t_count += 1
            else:
                save_path = curr_eval_dir
                e_count += 1
            img_name = f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(save_path / img_name), frame)

        frame_idx += 1

    cap.release()
    print(f"  [完成] {video_name}: Train({t_count}), Eval({e_count})")
    return "ok", video_path, t_count, e_count


def extract_frames(source_root, fps=60, split_ratio=0.8, num_workers=16):
    log_path = BASE_DIR / "log.txt"
    failed_log_path = BASE_DIR / "failed_videos.txt"

    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    video_extensions = (".mp4", ".avi", ".mov", ".mkv")

    tasks = []
    for root, _, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, source_root)
                video_name = os.path.splitext(file)[0]
                tasks.append((video_path, rel_path, video_name, fps, split_ratio))

    print(f"共找到 {len(tasks)} 个视频，使用 {num_workers} 个进程处理...")
    start_time = datetime.datetime.now()

    total = skipped = failed = train_frames = eval_frames = 0
    failed_videos = []

    with Pool(processes=num_workers) as pool:
        for status, path, t, e in pool.imap_unordered(process_one_video, tasks):
            if status == "skipped":
                skipped += 1
            elif status == "failed":
                failed += 1
                failed_videos.append(path)
            else:
                total += 1
                train_frames += t
                eval_frames += e

    duration = datetime.datetime.now() - start_time
    report = (
        f"\n{'='*50}\n"
        f"处理完成报告\n"
        f"{'='*50}\n"
        f"生成路径: {BASE_DIR}\n"
        f"总耗时: {duration}\n"
        f"处理视频总数: {total}\n"
        f"跳过视频数(已存在): {skipped}\n"
        f"失败视频数: {failed}\n"
        f"Train 帧数总计: {train_frames}\n"
        f"Eval 帧数总计: {eval_frames}\n"
        f"总计帧数: {train_frames + eval_frames}\n"
        f"{'='*50}\n"
    )
    print(report)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(report)

    if failed_videos:
        with open(failed_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_videos) + "\n")
        print(f"失败视频列表已保存至: {failed_log_path}")


if __name__ == "__main__":
    source_video_dir = "/mnt/nas/datasets4/open-p2p/cuphead"

    if not os.path.exists(source_video_dir):
        print(f"错误: 找不到源视频目录 '{source_video_dir}'")
    else:
        extract_frames(source_video_dir, fps=60, split_ratio=0.8, num_workers=16)
