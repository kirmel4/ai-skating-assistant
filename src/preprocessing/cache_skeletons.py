from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML, display

from src.preprocessing.skeleton import preprocess_skeleton


POSE_EDGES = [
    # face / head
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),

    # shoulders
    (11, 12),

    # left arm
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),

    # right arm
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),

    # torso
    (11, 23), (12, 24), (23, 24),

    # left leg
    (23, 25), (25, 27),
    (27, 29), (27, 31), (29, 31),

    # right leg
    (24, 26), (26, 28),
    (28, 30), (28, 32), (30, 32),
]


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_batched_skeletons(skeletons):

    skeletons = to_numpy(skeletons)

    if skeletons.ndim == 3:
        skeletons = skeletons[None, ...]

    if skeletons.ndim != 4:
        raise ValueError(
            f"Expected skeletons shape [B, T, V, C] or [T, V, C], got {skeletons.shape}"
        )

    B, T, V, C = skeletons.shape

    if V != 33:
        raise ValueError(f"Expected 33 landmarks, got {V}")

    if C < 4:
        raise ValueError(
            f"Expected at least 4 channels: x, y, z, visibility. Got {C}"
        )

    return skeletons


def get_xy_limits(
    skeleton,
    min_visibility: float = 0.3,
    margin: float = 0.2,
    fallback_xlim=(-1.0, 1.0),
    fallback_ylim=(1.0, -1.0),
):
    skeleton = to_numpy(skeleton)

    x = skeleton[..., 0]
    y = skeleton[..., 1]
    visibility = skeleton[..., 3]

    finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(visibility)
    visible_mask = visibility >= min_visibility
    mask = finite_mask & visible_mask

    if not np.any(mask):
        return fallback_xlim, fallback_ylim

    x_valid = x[mask]
    y_valid = y[mask]

    x_min = float(np.min(x_valid))
    x_max = float(np.max(x_valid))
    y_min = float(np.min(y_valid))
    y_max = float(np.max(y_valid))

    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    x_pad = x_range * margin
    y_pad = y_range * margin

    xlim = (x_min - x_pad, x_max + x_pad)

    # Инверсия y, чтобы визуально было как на изображении:
    # меньший y выше, больший y ниже.
    ylim = (y_max + y_pad, y_min - y_pad)

    return xlim, ylim


def draw_skeleton_frame(
    ax,
    skeleton,
    frame_idx: int,
    title: str | None = None,
    min_visibility: float = 0.3,
    xlim=None,
    ylim=None,
    show_axes: bool = True,
):
    frame = skeleton[frame_idx]

    x = frame[:, 0]
    y = frame[:, 1]
    visibility = frame[:, 3]

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(visibility)
    visible = finite & (visibility >= min_visibility)

    ax.clear()

    ax.scatter(x[visible], y[visible], s=30)

    for a, b in POSE_EDGES:
        if visible[a] and visible[b]:
            ax.plot(
                [x[a], x[b]],
                [y[a], y[b]],
                linewidth=2,
            )

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_aspect("equal", adjustable="box")

    if title is not None:
        ax.set_title(title)

    if show_axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
    else:
        ax.axis("off")


def animate_skeleton_batch(
    skeletons,
    min_visibility: float = 0.3,
    interval: int = 120,
    max_items: int | None = 4,
    margin: float = 0.2,
    show_axes: bool = True,
):

    skeletons = ensure_batched_skeletons(skeletons)

    if max_items is not None:
        skeletons = skeletons[:max_items]

    batch_size, num_frames, _, _ = skeletons.shape

    fig_width = max(4 * batch_size, 5)
    fig, axes = plt.subplots(
        1,
        batch_size,
        figsize=(fig_width, 5),
        squeeze=False,
    )
    axes = axes[0]

    # Важно: лимиты считаем один раз по всему ролику.
    # Иначе при анимации скелет будет прыгать из-за autoscale.
    limits = [
        get_xy_limits(
            skeletons[i],
            min_visibility=min_visibility,
            margin=margin,
        )
        for i in range(batch_size)
    ]

    def update(frame_idx):
        for batch_idx in range(batch_size):
            xlim, ylim = limits[batch_idx]

            draw_skeleton_frame(
                ax=axes[batch_idx],
                skeleton=skeletons[batch_idx],
                frame_idx=frame_idx,
                title=f"Sample {batch_idx} | Frame {frame_idx}",
                min_visibility=min_visibility,
                xlim=xlim,
                ylim=ylim,
                show_axes=show_axes,
            )

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=interval,
        repeat=True,
        cache_frame_data=False,
    )

    plt.close(fig)
    return HTML(anim.to_jshtml())


# @torch.no_grad()
def cache_skeleton_dataset(
    dataloader,
    pose_extractor_class,
    save_path: str,
    enable_preprocessing: bool = False,
    target_len: int = 96,
    pose_extractor_params: dict = None,
    debug_visualize: bool = False,
    debug_save_gif: bool = False,
    debug_output_dir: str = "outputs/skeleton_debug",
    debug_max_items: int = 4,
    debug_one_batch: bool = False,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    debug_output_dir = Path(debug_output_dir)
    if debug_save_gif:
        debug_output_dir.mkdir(parents=True, exist_ok=True)

    all_features = []
    all_jump_types = []
    all_rotations = []
    all_underrotations = []
    all_falls = []

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        frames = batch["frames"]
        jump_types = batch["jump_type"]
        rotations = batch["rotations"]
        underrotation = batch["underrotation"]
        fall = batch["fall"]

        # print(f"{frames.shape = }")
        batch_solved = []
        for video_frames in frames:
            # print(f"{video_frames.shape = }")
            # print(f"{pose_extractor_params = }")
            pose_extractor = pose_extractor_class(**pose_extractor_params)
            skeleton = pose_extractor(video_frames.unsqueeze(0))
            # print(f"{skeleton.shape = }")
            batch_solved.append(skeleton)
        
        skeleton = torch.cat(batch_solved, dim=0)
        # print(f"{skeleton.shape = }")

        features = (
            preprocess_skeleton(skeleton, target_len=target_len)
            if enable_preprocessing
            else skeleton
        )

        print(
            f"batch={batch_idx} | "
            f"skeleton.shape={tuple(skeleton.shape)} | "
            f"features.shape={tuple(features.shape)}"
        )

        if debug_visualize:
            print("Raw skeleton:")
            display(
                animate_skeleton_batch(
                    skeleton,
                    max_items=debug_max_items,
                    show_axes=True,
                )
            )

            print("Features:")
            display(
                animate_skeleton_batch(
                    features,
                    max_items=debug_max_items,
                    show_axes=True,
                )
            )

        all_features.append(features.cpu())
        all_jump_types.append(jump_types.cpu())
        all_rotations.append(rotations.cpu())
        all_underrotations.append(underrotation.cpu())
        all_falls.append(fall.cpu())

        if debug_one_batch:
            break

    data = {
        "features": torch.cat(all_features, dim=0),
        "jump_types": torch.cat(all_jump_types, dim=0),
        "rotations": torch.cat(all_rotations, dim=0),
        "underrotations": torch.cat(all_underrotations, dim=0),
        "falls": torch.cat(all_falls, dim=0),
    }

    torch.save(data, save_path)

    print(f"Saved cache: {save_path}")
    print(f"features: {tuple(data['features'].shape)}")
