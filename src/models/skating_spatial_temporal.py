from __future__ import annotations

import torch
import torch.nn as nn


def get_default_device() -> torch.device:
    """
    Apple Silicon: использует MPS, если доступен.
    Иначе CUDA, если доступна.
    Иначе CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


class JointAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        hidden = max(dim // 2, 8)

        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, V, D]
        returns: [B, T, D]
        """
        scores = self.scorer(x).squeeze(-1)       # [B, T, V]
        weights = torch.softmax(scores, dim=2)    # [B, T, V]

        return torch.sum(x * weights.unsqueeze(-1), dim=2)


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        hidden = max(dim // 2, 8)

        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        returns: [B, D]
        """
        scores = self.scorer(x).squeeze(-1)       # [B, T]
        weights = torch.softmax(scores, dim=1)    # [B, T]

        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class SpatialJointTransformer(nn.Module):
    """
    Spatial Transformer по точкам внутри каждого кадра.

    Вход:
        [B, T, V, D]

    Логика:
        для каждого кадра отдельно:
            V точек = sequence length
            D = feature dim
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim must be divisible by num_heads. Got dim={dim}, heads={num_heads}"
            )

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, V, D]
        returns: [B, T, V, D]
        """
        B, T, V, D = x.shape

        x = x.reshape(B * T, V, D).contiguous()
        x = self.encoder(x)
        x = x.reshape(B, T, V, D).contiguous()

        return x


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
    ):
        super().__init__()

        hidden = max(in_dim // 2, 32)

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SkatingSpatialTemporalMPSModel(nn.Module):
    """
    MPS-friendly модель:

        [B, T, 33, 7]
            ↓
        Linear projection каждой точки
            ↓
        joint_id embedding
            ↓
        Spatial Transformer по 33 точкам внутри каждого кадра
            ↓
        Joint attention pooling
            ↓
        BiLSTM по времени
            ↓
        Task-specific temporal attention
            ↓
        4 classification heads
    """

    def __init__(
        self,
        num_joints: int = 33,
        in_channels: int = 7,
        joint_dim: int = 48,
        hidden_size: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        spatial_layers: int = 1,
        spatial_heads: int = 4,
        num_jump_types: int = 6,
        num_rotations: int = 5,
        num_underrotation_classes: int = 2,
        num_fall_classes: int = 2,
    ):
        super().__init__()

        if joint_dim % spatial_heads != 0:
            raise ValueError(
                f"joint_dim must be divisible by spatial_heads. "
                f"Got joint_dim={joint_dim}, spatial_heads={spatial_heads}"
            )

        self.num_joints = num_joints
        self.in_channels = in_channels
        self.joint_dim = joint_dim

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, joint_dim),
            nn.GELU(),
        )

        self.joint_id_embedding = nn.Embedding(num_joints, joint_dim)

        self.spatial_encoder = SpatialJointTransformer(
            dim=joint_dim,
            num_layers=spatial_layers,
            num_heads=spatial_heads,
            dropout=dropout,
        )

        self.joint_pool = JointAttentionPool(joint_dim)

        lstm_dropout = dropout if lstm_layers > 1 else 0.0

        self.temporal_encoder = nn.LSTM(
            input_size=joint_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )

        encoded_size = hidden_size * 2

        self.jump_type_pool = TemporalAttentionPool(encoded_size)
        self.rotations_pool = TemporalAttentionPool(encoded_size)
        self.underrotation_pool = TemporalAttentionPool(encoded_size)
        self.fall_pool = TemporalAttentionPool(encoded_size)

        self.jump_type_head = MLPHead(
            in_dim=encoded_size,
            out_dim=num_jump_types,
            dropout=dropout,
        )

        self.rotations_head = MLPHead(
            in_dim=encoded_size,
            out_dim=num_rotations,
            dropout=dropout,
        )

        self.underrotation_head = MLPHead(
            in_dim=encoded_size,
            out_dim=num_underrotation_classes,
            dropout=dropout,
        )

        self.fall_head = MLPHead(
            in_dim=encoded_size,
            out_dim=num_fall_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: [B, T, V, C]

        Expected:
            V = 33
            C = 7

        returns:
            {
                "jump_type_logits":     [B, 6],
                "rotations_logits":     [B, 5],
                "underrotation_logits": [B, 2],
                "fall_logits":          [B, 2],
            }
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x shape [B, T, V, C], got {tuple(x.shape)}")

        B, T, V, C = x.shape

        if V != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {V}")

        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")

        x = x.float()

        # [B, T, V, C] -> [B, T, V, D]
        x = self.input_proj(x)

        # joint identity
        joint_ids = torch.arange(V, device=x.device)
        joint_emb = self.joint_id_embedding(joint_ids)  # [V, D]

        x = x + joint_emb.view(1, 1, V, self.joint_dim)

        # spatial relations between joints
        x = self.spatial_encoder(x)  # [B, T, V, D]

        # aggregate joints into per-frame representation
        x = self.joint_pool(x)       # [B, T, D]

        # temporal dynamics
        out, _ = self.temporal_encoder(x)  # [B, T, 2H]

        jump_feat = self.jump_type_pool(out)
        rot_feat = self.rotations_pool(out)
        ur_feat = self.underrotation_pool(out)
        fall_feat = self.fall_pool(out)

        return {
            "jump_type_logits": self.jump_type_head(jump_feat),
            "rotations_logits": self.rotations_head(rot_feat),
            "underrotation_logits": self.underrotation_head(ur_feat),
            "fall_logits": self.fall_head(fall_feat),
        }
