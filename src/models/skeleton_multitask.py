import torch
import torch.nn as nn


class SkatingMultiTaskModel(nn.Module):
    def __init__(
        self,
        num_joints: int = 33,
        in_channels: int = 7,
        hidden_size: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_size = num_joints * in_channels

        self.encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        encoded_size = hidden_size * 2

        self.jump_type_head = nn.Linear(
            encoded_size,
            6
        )

        self.rotations_head = nn.Linear(
            encoded_size,
            5,
        )

        self.underrotation_head = nn.Linear(
            encoded_size,
            2,
        )

        self.fall_head = nn.Linear(
            encoded_size,
            2,
        )

    def forward(self, x: torch.Tensor) -> dict:
        B, T, V, C = x.shape
        # print(f"initial {x.shape = }")
        x = x.reshape(B, T, V * C)
        # print(f"new {x.shape = }")

        out, _ = self.encoder(x)
        # print(f"{out.shape = }")
        pooled = out.mean(dim=1)
        # print(f"{pooled.shape = }")

        output = {
            "jump_type_logits": self.jump_type_head(pooled),
            "rotations_logits": self.rotations_head(pooled),
            "underrotation_logits": self.underrotation_head(pooled),
            "fall_logits": self.fall_head(pooled),
        }

        # print(f"{output = }")

        return output