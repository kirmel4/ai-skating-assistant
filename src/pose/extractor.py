from typing import Literal

import torch
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipePoseLandmarkerExtractor:
    def __init__(
        self,
        model_path: str,
        fps: int = 30,
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_world_landmarks: bool = False,
        delegate: Literal["cpu", "gpu"] = "cpu",
    ):
        self.fps = fps
        self.frame_idx = 0
        self.use_world_landmarks = use_world_landmarks

        if delegate == "gpu":
            mp_delegate = python.BaseOptions.Delegate.GPU
        else:
            mp_delegate = python.BaseOptions.Delegate.CPU

        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=mp_delegate,
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def close(self):
        self.landmarker.close()

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        assert frames.ndim == 5, f"Expected [B,T,C,H,W], got {frames.shape}"

        # print(f"{frames.shape = }")

        frames = frames.detach().cpu()

        if frames.max() <= 1.0:
            frames = frames * 255.0

        frames = frames.clamp(0, 255).byte()

        B, T, C, H, W = frames.shape
        output = torch.zeros(B, T, 33, 4, dtype=torch.float32)

        for b in range(B):
            for t in range(T):
                img = frames[b, t].permute(1, 2, 0).numpy()
                img = np.ascontiguousarray(img)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=img,
                )

                timestamp_ms = int(self.frame_idx * 1000 / self.fps)
                self.frame_idx += 1

                result = self.landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
                # result = self.landmarker.detect(
                #     mp_image,
                # )

                # print(f"{result = }")

                if not result.pose_landmarks:
                    print("No landmarks detected!")
                    continue

                if self.use_world_landmarks and result.pose_world_landmarks:
                    landmarks = result.pose_world_landmarks[0]
                else:
                    landmarks = result.pose_landmarks[0]

                for i, lm in enumerate(landmarks):
                    output[b, t, i, 0] = lm.x
                    output[b, t, i, 1] = lm.y
                    output[b, t, i, 2] = lm.z
                    output[b, t, i, 3] = lm.visibility

        # print(f"{output.shape = }")

        return output
    

