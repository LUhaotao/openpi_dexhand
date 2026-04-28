"""
Script to convert HDF5 robot data folders to a single LeRobot dataset v2.0 format.
Supports both the original Aloha subfolder layout and flat episode folders such as
franka + xhand teleoperation data.

Usage:
    python script/convert_tidy_up_to_lerobot.py --repo-id tidy_up_v4
    python script/convert_tidy_up_to_lerobot.py --repo-id tidy_up_v4 --resume
"""

from __future__ import annotations

import dataclasses
import json
import logging
import signal
import sys
import gc
from collections import defaultdict
from pathlib import Path
import shutil
from typing import Literal

import av
import h5py
from lerobot.common.datasets.compute_stats import (
    auto_downsample_height_width,
    get_feature_stats,
    sample_indices,
)
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    check_timestamps_sync,
    get_episode_data_index,
)
import numpy as np
import tqdm
import tyro

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
    ]
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
_interrupted = False

# 配置
LEROBOT_HOME = Path("/home/aloha/data/real_data/scene_0003")
SOURCE_DIR = Path("/home/aloha/data/real_data/scene_0003/real")
SUBFOLDERS = ["Blue_bowl", "carrot", "eggplant" , "red_chili"]  # 子文件夹列表
CAMERAS = ['cam_front', 'cam_left', 'cam_right', 'cam_high']  # 相机列表

ALOHA_MOTORS = [
    "right_waist", "right_shoulder", "right_elbow",
    "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper",
    "left_waist", "left_shoulder", "left_elbow",
    "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate", "left_gripper",
]

FRANKA_XHAND_MOTORS = [
    "franka_tcp_x", "franka_tcp_y", "franka_tcp_z",
    "franka_tcp_rx", "franka_tcp_ry", "franka_tcp_rz",
    "xhand_joint_00", "xhand_joint_01", "xhand_joint_02", "xhand_joint_03",
    "xhand_joint_04", "xhand_joint_05", "xhand_joint_06", "xhand_joint_07",
    "xhand_joint_08", "xhand_joint_09", "xhand_joint_10", "xhand_joint_11",
]


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _interrupted
    _interrupted = True
    logger.warning(f"\n收到中断信号 ({signum})，正在安全退出...")
    sys.exit(130)


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 8
    image_writer_threads: int = 4
    video_backend: str | None = None


@dataclasses.dataclass(frozen=True)
class VideoEncodingConfig:
    codec: Literal["h264", "hevc", "libsvtav1"] = "libsvtav1"
    pix_fmt: str = "yuv420p"
    gop: int | None = 2
    crf: int | None = 30
    fast_decode: int = 0


DEFAULT_DATASET_CONFIG = DatasetConfig()
DEFAULT_VIDEO_ENCODING_CONFIG = VideoEncodingConfig()


class StreamVideoWriter:
    """Write camera frames directly to mp4, avoiding the temporary PNG stage."""

    def __init__(self, video_path: Path, fps: int, encoding_config: VideoEncodingConfig) -> None:
        self.video_path = Path(video_path)
        self.fps = fps
        self.encoding_config = encoding_config
        self.output: av.container.OutputContainer | None = None
        self.stream: av.video.stream.VideoStream | None = None

    def _mux_packets(self, packets) -> None:
        if packets is None:
            return
        if not isinstance(packets, (list, tuple)):
            packets = [packets]
        for packet in packets:
            if packet is not None:
                self.output.mux(packet)

    def _ensure_open(self, frame: np.ndarray) -> None:
        if self.output is not None:
            return

        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        logging.getLogger("libav").setLevel(av.logging.ERROR)

        height, width = frame.shape[:2]
        video_options: dict[str, str] = {}
        if self.encoding_config.gop is not None:
            video_options["g"] = str(self.encoding_config.gop)
        if self.encoding_config.crf is not None:
            video_options["crf"] = str(self.encoding_config.crf)
        if self.encoding_config.fast_decode:
            key = "svtav1-params" if self.encoding_config.codec == "libsvtav1" else "tune"
            value = (
                f"fast-decode={self.encoding_config.fast_decode}"
                if self.encoding_config.codec == "libsvtav1"
                else "fastdecode"
            )
            video_options[key] = value

        self.output = av.open(str(self.video_path), "w")
        self.stream = self.output.add_stream(
            self.encoding_config.codec,
            self.fps,
            options=video_options,
        )
        self.stream.pix_fmt = self.encoding_config.pix_fmt
        self.stream.width = width
        self.stream.height = height

    def write(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 uint8 frame, got {frame.shape}")

        self._ensure_open(frame)
        video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
        self._mux_packets(self.stream.encode(video_frame))

    def close(self) -> None:
        if self.output is None:
            return

        self._mux_packets(self.stream.encode())
        self.output.close()
        av.logging.restore_default_callback()

        if not self.video_path.exists():
            raise OSError(f"Video encoding did not work. File not found: {self.video_path}.")

        self.output = None
        self.stream = None


def get_prompt_from_hdf5(file_path: Path, fallback: str = "unknown task") -> str:
    """从 HDF5 文件中读取 prompt 字段"""
    with h5py.File(file_path, "r") as f:
        if "prompt" in f:
            prompt = f["prompt"][()]
            if isinstance(prompt, bytes):
                prompt = prompt.decode('utf-8')
            prompt = str(prompt).strip()
            if prompt:
                return prompt
    return fallback


def collect_all_hdf5_files(
    source_dir: Path,
    subfolders: list[str] | None,
    default_task: str | None = None,
) -> list[tuple[Path, str]]:
    """
    收集 HDF5 文件。
    如果 subfolders 为空，则直接收集 source_dir/*.hdf5。
    
    Returns:
        list of (file_path, task_prompt) tuples
    """
    all_files = []

    if not subfolders:
        subfolders = ["."]

    for subfolder in subfolders:
        raw_dir = source_dir if subfolder in ["", "."] else source_dir / subfolder
        if not raw_dir.exists():
            logger.warning(f"目录不存在: {raw_dir}")
            continue
        
        hdf5_files = sorted(raw_dir.glob("*.hdf5"))
        if not hdf5_files:
            logger.warning(f"在 {raw_dir} 中没有找到 HDF5 文件")
            continue
        
        # 从第一个文件读取 prompt（假设同一子文件夹的所有文件 prompt 相同）
        fallback_task = default_task or (source_dir.parent.name if subfolder in ["", "."] else subfolder)
        task_prompt = get_prompt_from_hdf5(hdf5_files[0], fallback=fallback_task)
        logger.info(f"目录 {raw_dir}: {len(hdf5_files)} 个文件, task='{task_prompt}'")
        
        for f in hdf5_files:
            all_files.append((f, task_prompt))
    
    return all_files


def get_cameras_from_hdf5(file_path: Path) -> list[str]:
    """从 HDF5 文件获取相机列表"""
    with h5py.File(file_path, "r") as ep:
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]


def get_camera_shapes_from_hdf5(file_path: Path, cameras: list[str]) -> dict[str, tuple[int, int, int]]:
    """返回 LeRobot 使用的图像 shape: (channels, height, width)。"""
    camera_shapes = {}
    with h5py.File(file_path, "r") as ep:
        for camera in cameras:
            dataset = ep[f"/observations/images/{camera}"]
            if len(dataset.shape) != 4:
                raise ValueError(f"{camera} 图像应为 4 维 (T,H,W,C) 或 (T,C,H,W)，实际为 {dataset.shape}")

            frame_shape = tuple(dataset.shape[1:])
            if frame_shape[0] in (1, 3, 4):
                channels, height, width = frame_shape
            elif frame_shape[-1] in (1, 3, 4):
                height, width, channels = frame_shape
            else:
                raise ValueError(f"无法判断 {camera} 的通道维，shape={dataset.shape}")

            if channels != 3:
                raise ValueError(f"当前转换脚本只支持 RGB 3 通道图像，{camera} channels={channels}")

            camera_shapes[camera] = (channels, height, width)
    return camera_shapes


def infer_vector_dims(file_path: Path) -> tuple[int, int]:
    """推断 observation.state 和 action 维度。保留旧 Aloha 16->14 action 的兼容逻辑。"""
    with h5py.File(file_path, "r") as ep:
        state_dim = ep["/observations/qpos"].shape[1]
        raw_action_dim = ep["/action"].shape[1]

    if raw_action_dim == 16 and state_dim == 14:
        return state_dim, 14
    return state_dim, raw_action_dim


def get_vector_names(robot_type: str, dim: int) -> list[str]:
    if robot_type == "franka_xhand" and dim == len(FRANKA_XHAND_MOTORS):
        return FRANKA_XHAND_MOTORS
    if dim == len(ALOHA_MOTORS):
        return ALOHA_MOTORS
    return [f"dim_{i:02d}" for i in range(dim)]


def has_velocity(file_path: Path) -> bool:
    with h5py.File(file_path, "r") as ep:
        return "/observations/qvel" in ep


def has_effort(file_path: Path) -> bool:
    with h5py.File(file_path, "r") as ep:
        return "/observations/effort" in ep


def create_lerobot_dataset(
    repo_id: str,
    cameras: list[str],
    camera_shapes: dict[str, tuple[int, int, int]],
    state_dim: int,
    action_dim: int,
    mode: Literal["video", "image"] = "video",
    *,
    has_vel: bool = False,
    has_eff: bool = False,
    robot_type: str = "aloha",
    fps: int = 50,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    resume: bool = False,
    lerobot_home: Path = LEROBOT_HOME,
) -> LeRobotDataset:
    """创建或加载 LeRobot 数据集"""
    dataset_path = lerobot_home / repo_id
    image_writer_processes = dataset_config.image_writer_processes
    image_writer_threads = dataset_config.image_writer_threads
    if mode == "video":
        image_writer_processes = 0
        image_writer_threads = 0
    
    # 如果 resume=True 且数据集已存在，尝试加载
    if resume and dataset_path.exists():
        logger.info(f"尝试加载已有数据集: {dataset_path}")
        meta_info_path = dataset_path / "meta" / "info.json"
        if meta_info_path.exists():
            try:
                dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_path))
                logger.info(f"成功加载已有数据集，当前有 {dataset.num_episodes} 个 episodes")
                return dataset
            except Exception as e:
                logger.warning(f"加载已有数据集失败: {e}，将创建新数据集")
    
    state_names = get_vector_names(robot_type, state_dim)
    action_names = get_vector_names(robot_type, action_dim)
    
    # 定义特征
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [state_names],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [action_names],
        },
    }
    
    if has_vel:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [state_names],
        }
    
    if has_eff:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [state_names],
        }
    
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": camera_shapes[cam],
            "names": ["channels", "height", "width"],
        }
    
    # 如果不是 resume 模式，删除已存在的数据集
    if not resume and dataset_path.exists():
        logger.info(f"删除已存在的数据集: {dataset_path}")
        shutil.rmtree(dataset_path)
    
    # 创建新数据集
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        video_backend=dataset_config.video_backend,
        root=dataset_path
    )


def load_episode_arrays(ep: h5py.File):
    """加载单个 episode 的数据"""
    state = ep["/observations/qpos"][:].astype(np.float32, copy=False)
    action = ep["/action"][:].astype(np.float32, copy=False)

    # 兼容旧 Aloha 数据：16 维 action 中第 7 和 15 维是重复 gripper。
    if action.shape[1] == 16 and state.shape[1] == 14:
        action = action[:, [i for i in range(16) if i not in [7, 15]]]

    velocity = None
    if "/observations/qvel" in ep:
        velocity = ep["/observations/qvel"][:].astype(np.float32, copy=False)

    effort = None
    if "/observations/effort" in ep:
        effort = ep["/observations/effort"][:].astype(np.float32, copy=False)

    return state, action, velocity, effort


def image_frame_to_hwc_uint8(frame: np.ndarray, camera: str) -> np.ndarray:
    """Normalize an HDF5 image frame to HxWx3 uint8 for direct video encoding."""
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"{camera}: expected 3D image frame, got shape {frame.shape}")

    if frame.shape[-1] == 3:
        image = frame
    elif frame.shape[0] == 3:
        image = np.moveaxis(frame, 0, -1)
    else:
        raise ValueError(f"{camera}: expected HxWx3 or 3xHxW image frame, got shape {frame.shape}")

    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return np.ascontiguousarray(image.astype(np.uint8))


def prepare_image_for_stats(image: np.ndarray) -> np.ndarray:
    channel_first = np.moveaxis(np.asarray(image, dtype=np.uint8), -1, 0)
    return auto_downsample_height_width(channel_first)


def compute_streaming_episode_stats(
    features: dict[str, dict],
    episode_buffer: dict[str, np.ndarray],
    sampled_visual_frames: dict[str, list[np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    ep_stats: dict[str, dict[str, np.ndarray]] = {}
    for key, ft in features.items():
        if ft["dtype"] == "string":
            continue

        if ft["dtype"] in ["image", "video"]:
            sampled_frames = sampled_visual_frames.get(key)
            if not sampled_frames or any(frame is None for frame in sampled_frames):
                raise ValueError(f"Missing sampled frames for visual feature: {key}")

            ep_ft_array = np.stack(sampled_frames, axis=0)
            stats = get_feature_stats(ep_ft_array, axis=(0, 2, 3), keepdims=True)
            ep_stats[key] = {
                name: value if name == "count" else np.squeeze(value / 255.0, axis=0)
                for name, value in stats.items()
            }
            continue

        data = episode_buffer[key]
        ep_stats[key] = get_feature_stats(data, axis=0, keepdims=data.ndim == 1)

    return ep_stats


def save_streaming_video_episode(
    dataset: LeRobotDataset,
    task_prompt: str,
    state: np.ndarray,
    action: np.ndarray,
    velocity: np.ndarray | None,
    effort: np.ndarray | None,
    sampled_visual_frames: dict[str, list[np.ndarray]],
) -> None:
    episode_index = dataset.meta.total_episodes
    episode_length = int(state.shape[0])

    frame_index = np.arange(episode_length, dtype=np.int64)
    timestamp = frame_index.astype(np.float32) / float(dataset.fps)
    episode_index_array = np.full((episode_length,), episode_index, dtype=np.int64)
    index = np.arange(
        dataset.meta.total_frames,
        dataset.meta.total_frames + episode_length,
        dtype=np.int64,
    )

    task_index = dataset.meta.get_task_index(task_prompt)
    if task_index is None:
        dataset.meta.add_task(task_prompt)
        task_index = dataset.meta.get_task_index(task_prompt)
    task_index_array = np.full((episode_length,), task_index, dtype=np.int64)

    episode_buffer: dict[str, np.ndarray] = {
        "observation.state": state,
        "action": action,
        "timestamp": timestamp,
        "frame_index": frame_index,
        "episode_index": episode_index_array,
        "index": index,
        "task_index": task_index_array,
    }
    if velocity is not None:
        episode_buffer["observation.velocity"] = velocity
    if effort is not None:
        episode_buffer["observation.effort"] = effort

    dataset._save_episode_table(episode_buffer, episode_index)

    ep_stats = compute_streaming_episode_stats(
        dataset.features,
        episode_buffer,
        sampled_visual_frames,
    )
    dataset.meta.save_episode(episode_index, episode_length, [task_prompt], ep_stats)

    ep_data_index = get_episode_data_index(dataset.meta.episodes, [episode_index])
    ep_data_index_np = {key: value.numpy() for key, value in ep_data_index.items()}
    check_timestamps_sync(
        timestamp,
        episode_index_array,
        ep_data_index_np,
        dataset.fps,
        dataset.tolerance_s,
    )
    dataset.episode_buffer = dataset.create_episode_buffer()


def add_episode_with_streaming_videos(
    dataset: LeRobotDataset,
    ep: h5py.File,
    cameras: list[str],
    task_prompt: str,
    state: np.ndarray,
    action: np.ndarray,
    velocity: np.ndarray | None,
    effort: np.ndarray | None,
) -> None:
    num_frames = int(state.shape[0])
    episode_index = dataset.meta.total_episodes
    sample_idx_list = sample_indices(num_frames)
    sample_positions: dict[int, list[int]] = defaultdict(list)
    for pos, sample_idx in enumerate(sample_idx_list):
        sample_positions[int(sample_idx)].append(pos)

    sampled_visual_frames = {
        f"observation.images.{camera}": [None] * len(sample_idx_list)
        for camera in cameras
    }
    writers = {
        f"observation.images.{camera}": StreamVideoWriter(
            dataset.root / dataset.meta.get_video_file_path(episode_index, f"observation.images.{camera}"),
            dataset.fps,
            DEFAULT_VIDEO_ENCODING_CONFIG,
        )
        for camera in cameras
    }

    try:
        for frame_idx in range(num_frames):
            sampled_positions = sample_positions.get(frame_idx)
            for camera in cameras:
                image = image_frame_to_hwc_uint8(ep[f"/observations/images/{camera}"][frame_idx], camera)
                video_key = f"observation.images.{camera}"
                writers[video_key].write(image)

                if sampled_positions:
                    sampled_frame = prepare_image_for_stats(image)
                    for pos in sampled_positions:
                        sampled_visual_frames[video_key][pos] = sampled_frame
    finally:
        for writer in writers.values():
            writer.close()

    save_streaming_video_episode(
        dataset,
        task_prompt,
        state,
        action,
        velocity,
        effort,
        sampled_visual_frames,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    all_files: list[tuple[Path, str]],
    cameras: list[str],
    start_idx: int = 0,
) -> LeRobotDataset:
    """填充数据集"""
    
    total_episodes = len(all_files) - start_idx
    logger.info(f"开始处理 {total_episodes} 个 episodes（从索引 {start_idx} 开始）")
    
    # 统计不同 task 的数量
    task_counts = {}
    for _, task in all_files[start_idx:]:
        task_counts[task] = task_counts.get(task, 0) + 1
    logger.info(f"Task 分布: {task_counts}")
    
    for ep_idx in tqdm.tqdm(range(start_idx, len(all_files)), desc="处理 episodes"):
        if _interrupted:
            logger.warning("处理被中断")
            break
        
        ep_path, task_prompt = all_files[ep_idx]
        
        try:
            with h5py.File(ep_path, "r") as ep:
                state, action, velocity, effort = load_episode_arrays(ep)
                num_frames = state.shape[0]

                if action.shape[0] != num_frames:
                    raise ValueError(f"action 帧数 {action.shape[0]} != state 帧数 {num_frames}")

                for camera in cameras:
                    camera_path = f"/observations/images/{camera}"
                    if camera_path not in ep:
                        raise ValueError(f"缺少相机数据: {camera_path}")
                    if ep[camera_path].shape[0] != num_frames:
                        raise ValueError(f"{camera} 帧数 {ep[camera_path].shape[0]} != state 帧数 {num_frames}")

                if dataset.meta.video_keys:
                    add_episode_with_streaming_videos(
                        dataset,
                        ep,
                        cameras,
                        task_prompt,
                        state,
                        action,
                        velocity,
                        effort,
                    )
                else:
                    for i in range(num_frames):
                        frame = {
                            "observation.state": state[i],
                            "action": action[i],
                            "task": task_prompt,  # 使用从 HDF5 读取的 task prompt
                        }

                        for camera in cameras:
                            frame[f"observation.images.{camera}"] = ep[f"/observations/images/{camera}"][i]

                        if velocity is not None:
                            frame["observation.velocity"] = velocity[i]
                        if effort is not None:
                            frame["observation.effort"] = effort[i]

                        dataset.add_frame(frame)

                    dataset.save_episode()
            
            # 清理内存
            del state, action
            if velocity is not None:
                del velocity
            if effort is not None:
                del effort
            gc.collect()
            
        except KeyboardInterrupt:
            logger.warning(f"在 episode {ep_idx} 处理时收到中断信号")
            raise
        except Exception as e:
            logger.error(f"处理 episode {ep_idx} ({ep_path.name}) 时出错: {e}", exc_info=True)
            logger.warning(f"跳过 episode {ep_idx}，继续处理下一个")
            continue
    
    logger.info(f"完成处理，共处理 {total_episodes} 个 episodes")
    return dataset


def convert_tidy_up_data(
    repo_id: str = "tidy_up_v4",
    source_dir: Path = SOURCE_DIR,
    subfolders: list[str] | None = SUBFOLDERS,
    *,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    resume: bool = False,
    start_from_episode: int | None = None,
    image_writer_processes: int = 8,
    image_writer_threads: int = 4,
    use_videos: bool = True,
    lerobot_home: Path = LEROBOT_HOME,
    robot_type: str = "aloha",
    fps: int = 50,
    default_task: str | None = None,
):
    """
    将 tidy_up 数据转换为 LeRobot 格式

    Args:
        repo_id: 数据集名称
        source_dir: 源数据目录
        subfolders: 子文件夹列表 (如 ['s1', 's2', 's3'])
        push_to_hub: 是否推送到 HuggingFace Hub
        mode: 'video' 或 'image'
        resume: 是否从断点续传
        start_from_episode: 从指定 episode 开始
        image_writer_processes: 图像写入进程数
        image_writer_threads: 图像写入线程数
        use_videos: 是否使用视频格式存储
        lerobot_home: LeRobot 数据集根目录
        robot_type: 机器人类型，写入 LeRobot meta/info.json
        fps: 数据集帧率
        default_task: 当 HDF5 中 prompt 为空时使用的 task
    """
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    dataset_config = DatasetConfig(
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )
    
    try:
        logger.info(f"="*60)
        logger.info(f"开始转换数据")
        logger.info(f"源目录: {source_dir}")
        logger.info(f"子文件夹: {subfolders}")
        logger.info(f"目标数据集: {repo_id}")
        logger.info(f"="*60)
        
        # 收集所有 HDF5 文件
        all_files = collect_all_hdf5_files(source_dir, subfolders, default_task=default_task)
        if not all_files:
            raise ValueError("没有找到任何 HDF5 文件")
        
        logger.info(f"总共找到 {len(all_files)} 个 HDF5 文件")
        
        # 获取相机列表（从第一个文件）
        cameras = get_cameras_from_hdf5(all_files[0][0])
        logger.info(f"检测到相机: {cameras}")
        camera_shapes = get_camera_shapes_from_hdf5(all_files[0][0], cameras)
        logger.info(f"相机 shape: {camera_shapes}")

        state_dim, action_dim = infer_vector_dims(all_files[0][0])
        logger.info(f"state_dim: {state_dim}, action_dim: {action_dim}, robot_type: {robot_type}, fps: {fps}")
        
        # 检查 velocity 和 effort
        has_vel = has_velocity(all_files[0][0])
        has_eff = has_effort(all_files[0][0])
        logger.info(f"has_velocity: {has_vel}, has_effort: {has_eff}")
        
        # 确定起始索引
        start_idx = 0
        dataset_path = lerobot_home / repo_id

        if start_from_episode is not None:
            start_idx = start_from_episode
            logger.info(f"从指定的 episode {start_idx} 开始处理")
        elif resume and dataset_path.exists():
            # 自动检测已处理的 episode 数量
            meta_info_path = dataset_path / "meta" / "info.json"
            if meta_info_path.exists():
                with open(meta_info_path, "r") as f:
                    info = json.load(f)
                    start_idx = info.get("total_episodes", 0)
                    logger.info(f"检测到已有 {start_idx} 个 episodes，从 episode {start_idx} 开始续传")
        
        if start_idx >= len(all_files):
            logger.info(f"所有 {len(all_files)} 个 episodes 已处理完成，无需继续")
            return
        
        # 创建或加载数据集
        dataset = create_lerobot_dataset(
            repo_id=repo_id,
            cameras=cameras,
            camera_shapes=camera_shapes,
            state_dim=state_dim,
            action_dim=action_dim,
            mode=mode,
            has_vel=has_vel,
            has_eff=has_eff,
            robot_type=robot_type,
            fps=fps,
            dataset_config=dataset_config,
            resume=resume or (start_idx > 0),
            lerobot_home=lerobot_home,
        )
        logger.info(f"数据集已{'加载' if resume else '创建'}: {lerobot_home / repo_id}")
        
        # 填充数据
        dataset = populate_dataset(
            dataset,
            all_files,
            cameras=cameras,
            start_idx=start_idx,
        )
        
        # 推送到 Hub
        if push_to_hub:
            logger.info("推送数据集到 Hub...")
            dataset.push_to_hub()
            logger.info("数据集已推送到 Hub")
        
        logger.info("="*60)
        logger.info("数据转换完成！")
        logger.info(f"数据集位置: {lerobot_home / repo_id}")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.warning("程序被用户中断")
        sys.exit(130)
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    tyro.cli(convert_tidy_up_data)
