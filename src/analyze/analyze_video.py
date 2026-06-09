# src/inference/analyze_video.py

import torch

from src.pose.extractor import MediaPipePoseExtractor
from src.preprocessing.skeleton import preprocess_skeleton


@torch.no_grad()
def analyze_batch(
    model,
    frames,
    element_names,
    error_names,
    device="cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    pose_extractor = MediaPipePoseExtractor()

    skeleton = pose_extractor(frames)
    features = preprocess_skeleton(skeleton).to(device)

    out = model(features)

    element_probs = torch.softmax(out["element_logits"], dim=1)
    error_probs = torch.sigmoid(out["error_logits"])
    quality = out["quality"]

    results = []

    for i in range(frames.shape[0]):
        element_id = element_probs[i].argmax().item()

        errors = {
            error_names[j]: float(error_probs[i, j].cpu())
            for j in range(len(error_names))
        }

        results.append({
            "element": element_names[element_id],
            "confidence": float(element_probs[i, element_id].cpu()),
            "quality_score": float(quality[i].cpu()),
            "errors": errors,
        })

    return results