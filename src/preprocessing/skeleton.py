import torch
import torch.nn.functional as F
import numpy as np


LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12



LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def interpolate_missing_landmarks(
    skeleton: torch.Tensor,
    min_visibility: float = 0.3,
    fill_value: float = 0.0,
) -> torch.Tensor:
    if skeleton.ndim != 4:
        raise ValueError(f"Expected [B, T, V, 4], got {skeleton.shape}")

    if skeleton.shape[-1] < 4:
        raise ValueError(f"Expected at least 4 channels, got {skeleton.shape[-1]}")

    device = skeleton.device
    dtype = skeleton.dtype

    skel = skeleton.detach().cpu().float().numpy().copy()

    B, T, V, C = skel.shape

    coords = skel[..., :3]
    visibility = skel[..., 3]

    finite_coords = np.isfinite(coords).all(axis=-1)
    finite_visibility = np.isfinite(visibility)

    non_zero = ~(
        np.isclose(coords[..., 0], 0.0)
        & np.isclose(coords[..., 1], 0.0)
        & np.isclose(coords[..., 2], 0.0)
        & np.isclose(visibility, 0.0)
    )

    valid = (
        finite_coords
        & finite_visibility
        & non_zero
        & (visibility >= min_visibility)
    )

    # Невалидным точкам visibility принудительно ставим 0.
    visibility[~valid] = 0.0

    time_idx = np.arange(T)

    for b in range(B):
        for v in range(V):
            valid_t = valid[b, :, v]

            # Если точка вообще ни разу не найдена в клипе,
            # заполняем координаты нулями. Visibility уже 0.
            if valid_t.sum() == 0:
                coords[b, :, v, :] = fill_value
                continue

            # Если точка найдена только один раз,
            # протягиваем эту координату на весь клип.
            if valid_t.sum() == 1:
                only_idx = np.where(valid_t)[0][0]
                coords[b, :, v, :] = coords[b, only_idx, v, :]
                continue

            # Обычная линейная интерполяция отдельно для x, y, z.
            valid_times = time_idx[valid_t]

            for c in range(3):
                valid_values = coords[b, valid_t, v, c]

                coords[b, :, v, c] = np.interp(
                    time_idx,
                    valid_times,
                    valid_values,
                )

    skel[..., :3] = coords
    skel[..., 3] = visibility

    return torch.tensor(skel, dtype=dtype, device=device)


def center_skeleton(skeleton: torch.Tensor) -> torch.Tensor:
    coords = skeleton[..., :3]
    vis = skeleton[..., 3:]

    pelvis = (coords[:, :, LEFT_HIP] + coords[:, :, RIGHT_HIP]) / 2.0
    coords = coords - pelvis[:, :, None, :]

    return torch.cat([coords, vis], dim=-1)


def scale_skeleton(skeleton: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    coords = skeleton[..., :3]
    vis = skeleton[..., 3:]

    shoulder_center = (coords[:, :, LEFT_SHOULDER] + coords[:, :, RIGHT_SHOULDER]) / 2.0
    hip_center = (coords[:, :, LEFT_HIP] + coords[:, :, RIGHT_HIP]) / 2.0

    body_scale = torch.norm(shoulder_center - hip_center, dim=-1)
    body_scale = body_scale.clamp_min(eps)

    coords = coords / body_scale[:, :, None, None]

    return torch.cat([coords, vis], dim=-1)


def add_velocity(skeleton: torch.Tensor) -> torch.Tensor:
    coords = skeleton[..., :3]
    velocity = torch.zeros_like(coords)
    velocity[:, 1:] = coords[:, 1:] - coords[:, :-1]

    return torch.cat([skeleton, velocity], dim=-1)


def resample_temporal(x: torch.Tensor, target_len: int = 96) -> torch.Tensor:
    B, T, V, C = x.shape

    x = x.permute(0, 2, 3, 1)
    x = x.reshape(B, V * C, T)

    x = F.interpolate(
        x,
        size=target_len,
        mode="linear",
        align_corners=False,
    )

    x = x.reshape(B, V, C, target_len)
    x = x.permute(0, 3, 1, 2) 
    return x


def preprocess_skeleton(
    skeleton: torch.Tensor,
    target_len: int = 96,
    min_visibility: float = 0.3,
    interpolate_missing: bool = True,
) -> torch.Tensor:
    if interpolate_missing:
        skeleton = interpolate_missing_landmarks(
            skeleton,
            min_visibility=min_visibility,
        )

    skeleton = center_skeleton(skeleton)
    skeleton = scale_skeleton(skeleton)
    skeleton = resample_temporal(skeleton, target_len)
    skeleton = add_velocity(skeleton)

    return skeleton