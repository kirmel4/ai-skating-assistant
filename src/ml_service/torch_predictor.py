from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from src.ml_service.contracts import RawPrediction


class TorchMultitaskPredictor:
    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: Path,
        device: torch.device,
        normalize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.normalize_fn = normalize_fn

        self._load_checkpoint(checkpoint_path)

        self.model.eval()

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=True)

    @torch.inference_mode()
    def predict(
        self,
        frames: torch.Tensor,
        duration_sec: float,
    ) -> RawPrediction:
        frames = frames.to(self.device)

        if self.normalize_fn is not None:
            frames = self.normalize_fn(frames)

        output = self.model(frames)

        jump_logits, rotation_logits, underrotation_logits, fall_logits, rotations_value = (
            self._unpack_output(output)
        )

        jump_type_id = int(jump_logits.argmax(dim=1).item())
        rotation_id = int(rotation_logits.argmax(dim=1).item())
        underrotation_id = int(underrotation_logits.argmax(dim=1).item())
        fall_id = int(fall_logits.argmax(dim=1).item())

        return RawPrediction(
            jump_type_id=jump_type_id,
            rotation_id=rotation_id,
            underrotation_id=underrotation_id,
            fall_id=fall_id,
            rotations_value=rotations_value,
        )

    def _unpack_output(
        self,
        output,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float | None]:
        rotations_value = None

        if isinstance(output, dict):
            jump_logits = output["jump_type_logits"]
            rotation_logits = output["rotations_logits"]
            underrotation_logits = output["underrotation_logits"]
            fall_logits = output["fall_logits"]

            if "rotations_value" in output:
                rotations_value = float(output["rotations_value"].reshape(-1)[0].item())
            elif "rot_regression" in output:
                rotations_value = float(output["rot_regression"].reshape(-1)[0].item())

            return (
                jump_logits,
                rotation_logits,
                underrotation_logits,
                fall_logits,
                rotations_value,
            )

        if isinstance(output, tuple) or isinstance(output, list):
            if len(output) < 4:
                raise ValueError(
                    "Model output must contain at least 4 tensors: "
                    "jump, rotation, underrotation, fall"
                )

            jump_logits = output[0]
            rotation_logits = output[1]
            underrotation_logits = output[2]
            fall_logits = output[3]

            if len(output) >= 5 and output[4] is not None:
                rotations_value = float(output[4].reshape(-1)[0].item())

            return (
                jump_logits,
                rotation_logits,
                underrotation_logits,
                fall_logits,
                rotations_value,
            )

        raise TypeError(f"Unsupported model output type: {type(output)}")
    
