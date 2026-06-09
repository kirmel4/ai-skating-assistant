from dataclasses import dataclass


@dataclass(frozen=True)
class VideoConfig:
    num_frames: int = 32
    target_fps: float = 25.0
    image_size: int = 1024
    return_meta: bool = True


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 1
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    seed: int = 420