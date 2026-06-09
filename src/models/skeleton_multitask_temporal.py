import torch
import torch.nn as nn


class TemporalConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length")

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm1 = nn.LayerNorm(channels)

        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm2 = nn.LayerNorm(channels)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]

        returns:
            x: [B, T, C]
        """
        residual = x

        x = x.transpose(1, 2)    # [B, C, T]
        x = self.conv1(x)        # [B, C, T]
        x = x.transpose(1, 2)    # [B, T, C]

        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)    # [B, C, T]
        x = self.conv2(x)        # [B, C, T]
        x = x.transpose(1, 2)    # [B, T, C]

        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x + residual


class AttentionPooling(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]

        returns:
            pooled: [B, C]
        """
        scores = self.attention(x)          # [B, T, 1]
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1)   # [B, C]

        return pooled


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SkatingMultiTaskModel(nn.Module):
    def __init__(
        self,
        num_joints: int = 33,
        in_channels: int = 7,
        frame_embed_size: int = 256,
        hidden_size: int = 256,
        shared_size: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_joints = num_joints
        self.in_channels = in_channels
        self.input_size = num_joints * in_channels

        self.input_norm = nn.LayerNorm(self.input_size)

        self.frame_encoder = nn.Sequential(
            nn.Linear(self.input_size, frame_embed_size),
            nn.LayerNorm(frame_embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.temporal_conv = TemporalConvBlock(
            channels=frame_embed_size,
            kernel_size=5,
            dropout=dropout,
        )

        self.encoder = nn.LSTM(
            input_size=frame_embed_size,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        encoded_size = hidden_size * 2

        self.pooling = AttentionPooling(
            input_size=encoded_size,
            hidden_size=128,
        )

        self.shared_head = nn.Sequential(
            nn.Linear(encoded_size, shared_size),
            nn.LayerNorm(shared_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.jump_type_head = ClassificationHead(
            input_size=shared_size,
            hidden_size=shared_size // 2,
            num_classes=6,
            dropout=dropout,
        )

        self.rotations_head = ClassificationHead(
            input_size=shared_size,
            hidden_size=shared_size // 2,
            num_classes=5,
            dropout=dropout,
        )

        self.underrotation_head = ClassificationHead(
            input_size=shared_size,
            hidden_size=shared_size // 2,
            num_classes=2,
            dropout=dropout,
        )

        self.fall_head = ClassificationHead(
            input_size=shared_size,
            hidden_size=shared_size // 2,
            num_classes=2,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: [B, T, V, C]

        returns:
            jump_type_logits:      [B, 6]
            rotations_logits:      [B, 5]
            underrotation_logits:  [B, 2]
            fall_logits:           [B, 2]
        """
        B, T, V, C = x.shape

        if V != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {V}")

        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")

        x = x.reshape(B, T, V * C)       # [B, T, V*C]

        x = self.input_norm(x)
        x = self.frame_encoder(x)        # [B, T, frame_embed_size]

        x = self.temporal_conv(x)        # [B, T, frame_embed_size]

        out, _ = self.encoder(x)         # [B, T, hidden_size * 2]

        pooled = self.pooling(out)       # [B, hidden_size * 2]

        features = self.shared_head(pooled)

        return {
            "jump_type_logits": self.jump_type_head(features),
            "rotations_logits": self.rotations_head(features),
            "underrotation_logits": self.underrotation_head(features),
            "fall_logits": self.fall_head(features),
        }
