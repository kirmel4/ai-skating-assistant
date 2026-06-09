import torch
import torch.nn as nn


class SkeletonLSTMClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_joints: int = 33,
        in_channels: int = 7,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_size = num_joints * in_channels

        self.encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, V, C = x.shape
        x = x.reshape(B, T, V * C)

        out, _ = self.encoder(x)

        # mean pooling по времени
        out = out.mean(dim=1)

        logits = self.classifier(out)
        return logits