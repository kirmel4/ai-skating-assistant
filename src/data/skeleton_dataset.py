# src/data/skeleton_dataset.py

import torch
from torch.utils.data import Dataset


class CachedSkeletonDataset(Dataset):
    def __init__(self, path: str):
        self.data = torch.load(path, map_location="cpu")

    def __len__(self):
        return self.data["features"].shape[0]

    def __getitem__(self, idx):
        return {
            "features": self.data["features"][idx],
            "jump_types": self.data["jump_types"][idx],
            "rotations": self.data["rotations"][idx],
            "underrotations": self.data["underrotations"][idx],
            "falls": self.data["falls"][idx],
        }
