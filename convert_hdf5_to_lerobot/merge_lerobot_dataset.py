#!/usr/bin/env python3
"""
合并多个 LeRobot v2.1 数据集为一个。

用法:
    python merge_lerobot_datasets.py         --input_dir /home/luhaotao/data/aloha_motion_gen/data/scene_v2         --output_dir /home/luhaotao/data/aloha_motion_gen/data/merged

会递归扫描 input_dir 下所有 LeRobot 数据集根目录。
兼容旧的 lerobot_data 布局，以及新的 <scene>/lerobot/<repo_id> 布局。
"""

import argparse
import copy
import json
import os
import shutil

import pandas as pd


def is_dataset_root(path):
    required_files = [
        os.path.join(path, "meta", "info.json"),
        os.path.join(path, "meta", "episodes.jsonl"),
        os.path.join(path, "meta", "tasks.jsonl"),
    ]
    return all(os.path.isfile(file_path) for file_path in required_files)


def find_datasets(input_dir):
    """递归扫描 input_dir 下所有 LeRobot 数据集目录。"""
    datasets = []
    input_dir = os.path.abspath(input_dir)
    seen_roots = set()

    for current_dir, dirnames, _ in os.walk(input_dir):
        dirnames.sort()

        if not is_dataset_root(current_dir):
            continue

        dataset_root = os.path.abspath(current_dir)
        if dataset_root in seen_roots:
            dirnames[:] = []
            continue

        seen_roots.add(dataset_root)
        rel_name = os.path.relpath(dataset_root, input_dir)
        datasets.append({
            "name": "." if rel_name == "." else rel_name,
            "root": dataset_root,
        })

        # 当前目录已经是数据集根目录，不再继续向下扫描，避免重复命中其子目录。
        dirnames[:] = []

    datasets.sort(key=lambda item: item["name"])
    return datasets


def load_info(dataset_root):
    with open(os.path.join(dataset_root, "meta", "info.json"), "r") as f:
        return json.load(f)


def get_chunk_size(info, dataset_name):
    chunk_size = int(info.get("chunks_size", 1000))
    if chunk_size <= 0:
        raise ValueError(f"数据集 {dataset_name} 的 chunks_size 非法: {chunk_size}")
    return chunk_size


def validate_compatible_info(reference_info, candidate_info, dataset_name):
    mismatches = []

    for field in ["codebase_version", "robot_type", "fps", "data_path", "video_path"]:
        if reference_info.get(field) != candidate_info.get(field):
            mismatches.append(
                f"{field}: {reference_info.get(field)!r} != {candidate_info.get(field)!r}"
            )

    if reference_info.get("features") != candidate_info.get("features"):
        mismatches.append("features 不一致")

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(f"数据集 {dataset_name} 与首个数据集的元数据不兼容: {mismatch_text}")


def load_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items, path):
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def summarize_scalar_series(series):
    numeric_series = series.astype("float64")
    return {
        "min": [series.min().item()],
        "max": [series.max().item()],
        "mean": [float(numeric_series.mean())],
        "std": [float(numeric_series.std(ddof=0))],
        "count": [int(len(series))],
    }


def remap_task_indices(series, task_index_map, dataset_name, episode_index):
    mapped = series.map(task_index_map)
    if mapped.isna().any():
        missing_indices = sorted(set(series[mapped.isna()].tolist()))
        raise ValueError(
            f"数据集 {dataset_name} 的 episode {episode_index} 存在未映射的 task_index: {missing_indices}"
        )
    return mapped.astype("int64")


def update_episode_stats(source_stat, new_episode_index, df):
    updated_stat = copy.deepcopy(source_stat)
    updated_stat["episode_index"] = new_episode_index
    updated_stat.setdefault("stats", {})
    updated_stat["stats"]["episode_index"] = summarize_scalar_series(df["episode_index"])
    updated_stat["stats"]["index"] = summarize_scalar_series(df["index"])
    if "task_index" in df.columns:
        updated_stat["stats"]["task_index"] = summarize_scalar_series(df["task_index"])
    else:
        updated_stat["stats"].pop("task_index", None)
    return updated_stat


def merge_datasets(input_dir, output_dir):
    datasets = find_datasets(input_dir)
    if not datasets:
        print(f"在 {input_dir} 下未找到任何 LeRobot 数据集")
        return

    for dataset in datasets:
        dataset["info"] = load_info(dataset["root"])
        dataset["chunks_size"] = get_chunk_size(dataset["info"], dataset["name"])

    print(f"找到 {len(datasets)} 个数据集:")
    for dataset in datasets:
        print(f"  - {dataset['name']} -> {dataset['root']}")

    os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data", "chunk-000"), exist_ok=True)

    # 用第一个数据集的 info.json 作为模板
    merged_info = copy.deepcopy(datasets[0]["info"])
    output_chunk_size = get_chunk_size(merged_info, datasets[0]["name"])

    for dataset in datasets[1:]:
        validate_compatible_info(merged_info, dataset["info"], dataset["name"])

    all_tasks = []       # {task: str, task_index: int}
    all_episodes = []    # {episode_index, tasks, length}
    all_stats = []       # {episode_index, stats}
    task_to_index = {}
    global_episode_idx = 0
    global_frame_idx = 0
    total_videos = 0

    for dataset in datasets:
        ds_name = dataset["name"]
        ds_path = dataset["root"]
        ds_chunk_size = dataset["chunks_size"]
        print(f"\n处理数据集: {ds_name}")

        # 加载元数据
        tasks = load_jsonl(os.path.join(ds_path, "meta", "tasks.jsonl"))
        episodes = load_jsonl(os.path.join(ds_path, "meta", "episodes.jsonl"))
        stats_by_episode = {}
        stats_path = os.path.join(ds_path, "meta", "episodes_stats.jsonl")
        if os.path.isfile(stats_path):
            stats_by_episode = {
                stat["episode_index"]: stat
                for stat in load_jsonl(stats_path)
            }

        # 建立 task 映射: 旧 task_index -> 新 task_index
        task_index_map = {}
        for t in tasks:
            task_str = t["task"]
            if task_str not in task_to_index:
                new_idx = len(task_to_index)
                task_to_index[task_str] = new_idx
                all_tasks.append({"task_index": new_idx, "task": task_str})
            task_index_map[t["task_index"]] = task_to_index[task_str]

        # 处理每个 episode
        for ep in episodes:
            old_ep_idx = ep["episode_index"]
            new_ep_idx = global_episode_idx
            chunk = new_ep_idx // output_chunk_size
            chunk_dir = f"chunk-{chunk:03d}"

            # 复制 parquet 数据，重新编号 index/episode_index/task_index
            old_chunk = old_ep_idx // ds_chunk_size
            old_parquet = os.path.join(ds_path, "data", f"chunk-{old_chunk:03d}", f"episode_{old_ep_idx:06d}.parquet")
            if not os.path.isfile(old_parquet):
                print(f"  警告: 找不到 {old_parquet}, 跳过")
                continue

            df = pd.read_parquet(old_parquet)
            df["episode_index"] = new_ep_idx
            df["index"] = range(global_frame_idx, global_frame_idx + len(df))
            if "task_index" in df.columns:
                df["task_index"] = remap_task_indices(
                    df["task_index"],
                    task_index_map,
                    ds_name,
                    old_ep_idx,
                )

            new_data_dir = os.path.join(output_dir, "data", chunk_dir)
            os.makedirs(new_data_dir, exist_ok=True)
            df.to_parquet(os.path.join(new_data_dir, f"episode_{new_ep_idx:06d}.parquet"), index=False)

            # 复制视频文件
            old_video_chunk_dir = os.path.join(ds_path, "videos", f"chunk-{old_chunk:03d}")
            if os.path.isdir(old_video_chunk_dir):
                for cam_name in os.listdir(old_video_chunk_dir):
                    old_video = os.path.join(old_video_chunk_dir, cam_name, f"episode_{old_ep_idx:06d}.mp4")
                    if os.path.isfile(old_video):
                        new_video_dir = os.path.join(output_dir, "videos", chunk_dir, cam_name)
                        os.makedirs(new_video_dir, exist_ok=True)
                        shutil.copy2(old_video, os.path.join(new_video_dir, f"episode_{new_ep_idx:06d}.mp4"))
                        total_videos += 1

            # 记录 episode
            all_episodes.append({
                "episode_index": new_ep_idx,
                "tasks": ep["tasks"],
                "length": ep["length"],
            })
            if old_ep_idx in stats_by_episode:
                all_stats.append(update_episode_stats(stats_by_episode[old_ep_idx], new_ep_idx, df))

            global_frame_idx += len(df)
            global_episode_idx += 1
            print(f"  episode {old_ep_idx} -> {new_ep_idx} ({len(df)} frames)")

    # 写入合并后的元数据
    merged_info["total_episodes"] = global_episode_idx
    merged_info["total_frames"] = global_frame_idx
    merged_info["total_tasks"] = len(all_tasks)
    merged_info["total_videos"] = total_videos
    merged_info["chunks_size"] = output_chunk_size
    merged_info["total_chunks"] = (global_episode_idx - 1) // output_chunk_size + 1 if global_episode_idx > 0 else 0
    merged_info["splits"] = {"train": f"0:{global_episode_idx}"}

    with open(os.path.join(output_dir, "meta", "info.json"), "w") as f:
        json.dump(merged_info, f, indent=4, ensure_ascii=False)

    save_jsonl(all_tasks, os.path.join(output_dir, "meta", "tasks.jsonl"))
    save_jsonl(all_episodes, os.path.join(output_dir, "meta", "episodes.jsonl"))
    save_jsonl(all_stats, os.path.join(output_dir, "meta", "episodes_stats.jsonl"))

    print(f"\n合并完成!")
    print(f"  总 episodes: {global_episode_idx}")
    print(f"  总 frames:   {global_frame_idx}")
    print(f"  总 tasks:    {len(all_tasks)}")
    print(f"  总 videos:   {total_videos}")
    print(f"  输出目录:    {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并多个 LeRobot v2.1 数据集")
    parser.add_argument("--input_dir", required=True, help="包含多个 LeRobot 数据集的目录")
    parser.add_argument("--output_dir", required=True, help="合并后的输出目录")
    args = parser.parse_args()
    merge_datasets(args.input_dir, args.output_dir)
