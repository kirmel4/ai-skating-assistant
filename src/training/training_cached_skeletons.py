import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.models.skeleton_multitask_temporal_spacial import SkatingMultiTaskModel
from src.models.skating_spatial_temporal import SkatingSpatialTemporalMPSModel, get_default_device



def compute_f1(preds, targets, num_classes):

    f1_per_class = []

    for cls in range(num_classes):
        tp = ((preds == cls) & (targets == cls)).sum().item()
        fp = ((preds == cls) & (targets != cls)).sum().item()
        fn = ((preds != cls) & (targets == cls)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_per_class.append(f1)

    return sum(f1_per_class) / len(f1_per_class)


def plot_history(history):
    epochs = list(range(len(history["jump_type_f1"])))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["jump_type_f1"], label="jump_type_f1")
    plt.plot(epochs, history["rotations_f1"], label="rotations_f1")
    plt.plot(epochs, history["underrotation_f1"], label="underrotation_f1")
    plt.plot(epochs, history["fall_f1"], label="fall_f1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")
    plt.title("Validation F1")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def make_class_weights(labels: torch.Tensor, num_classes: int):
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    return weights


def train_cached(
    train_loader,
    val_loader,
    model: torch.nn.Module = None,
    epochs: int = 30,
    lr: float = 1e-4,
    device: str = None,
    print_every: int = 1,
    plot_every: int = 1,
):
    # device = torch.device(device if torch.cuda.is_available() else "cpu")
    device = get_default_device() if not device else device
    print(f"{device = }")

    train_data = train_loader.dataset.data 

    jump_type_weights = make_class_weights(
        train_data["jump_types"], 
        num_classes=6,
    ).to(device)

    rotations_weights = make_class_weights(
        train_data["rotations"], 
        num_classes=5,
    ).to(device)

    underrotation_weights = make_class_weights(
        train_data["underrotations"], 
        num_classes=2,
    ).to(device)

    fall_weights = make_class_weights(
        train_data["falls"], 
        num_classes=2,
    ).to(device)

    loss_jump_type = nn.CrossEntropyLoss(weight=jump_type_weights)
    loss_rotations = nn.CrossEntropyLoss(weight=rotations_weights)
    loss_underrotation = nn.CrossEntropyLoss(weight=underrotation_weights)
    loss_fall = nn.CrossEntropyLoss(weight=fall_weights)

    # model = model or SkatingSpatialTemporalMPSModel()

    # model = model or SkatingSpatialTemporalMPSModel(
    #     num_joints=33,
    #     in_channels=7,
    #     joint_dim=64,
    #     hidden_size=256,
    #     dropout=0.3,
    #     spatial_layers=2,
    #     spatial_heads=4,
    #     num_fall_classes=2,
    # )

    # model = model or SkatingMultiTaskModel(
    #     num_joints = 33,
    #     in_channels = 7,
    #     frame_embed_size = 256,
    #     hidden_size = 256,
    #     shared_size = 256,
    #     dropout = 0.3,
    # )


    # model = model or SkatingMultiTaskModel(
    #     num_joints = 33,
    #     in_channels = 7,
    #     frame_embed_size = 256,
    #     hidden_size = 256,
    #     shared_size = 256,
    #     dropout = 0.3,
    # )

    model = model or SkatingMultiTaskModel(
        num_joints = 33,
        in_channels = 7,
        joint_embed_size = 64,
        frame_embed_size = 256,
        hidden_size = 256,
        shared_size = 256,
        dropout = 0.3,
    )
    

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    history = {
        "train_loss": [],
        "jump_type_f1": [],
        "rotations_f1": [],
        "underrotation_f1": [],
        "fall_f1": [],
    }

    metrics = evaluate_cached(model, val_loader, device)

    history["train_loss"].append(None)
    history["jump_type_f1"].append(metrics["jump_type_f1"])
    history["rotations_f1"].append(metrics["rotations_f1"])
    history["underrotation_f1"].append(metrics["underrotation_f1"])
    history["fall_f1"].append(metrics["fall_f1"])

    print(
        f"Epoch 0/{epochs} | "
        f"jump_f1={metrics['jump_type_f1']:.4f} | "
        f"rot_f1={metrics['rotations_f1']:.4f} | "
        f"ur_f1={metrics['underrotation_f1']:.4f} | "
        f"fall_f1={metrics['fall_f1']:.4f}"
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader):
            x = batch["features"].float().to(device)

            y_jump_type = batch["jump_types"].long().to(device)
            y_rotations = batch["rotations"].long().to(device)
            y_underrotation = batch["underrotations"].long().to(device)
            y_fall = batch["falls"].long().to(device)


            # print(f"{y_fall = }")

            out = model(x)
            # print(f"Train: {y_fall = }")
            # print(f"Train: {out['fall_logits'] = }")

            loss = (
                loss_jump_type(out["jump_type_logits"], y_jump_type) +
                loss_rotations(out["rotations_logits"], y_rotations)
                # loss_underrotation(out["underrotation_logits"], y_underrotation) + 
                # loss_fall(out["fall_logits"], y_fall)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        metrics = evaluate_cached(model, val_loader, device)

        history["train_loss"].append(avg_loss)
        history["jump_type_f1"].append(metrics["jump_type_f1"])
        history["rotations_f1"].append(metrics["rotations_f1"])
        history["underrotation_f1"].append(metrics["underrotation_f1"])
        history["fall_f1"].append(metrics["fall_f1"])

        if print_every and (epoch + 1) % print_every == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"loss={avg_loss:.4f} | "
                f"jump_f1={metrics['jump_type_f1']:.4f} | "
                f"rot_f1={metrics['rotations_f1']:.4f} | "
                f"ur_f1={metrics['underrotation_f1']:.4f} | "
                f"fall_f1={metrics['fall_f1']:.4f}"
            )

        if plot_every and (epoch + 1) % plot_every == 0:
            plot_history(history)

    plot_history(history)

    return model, history


@torch.no_grad()
def evaluate_cached(model, loader, device):
    model.eval()

    preds = {
        "jump_types": [],
        "rotations": [],
        "underrotations": [],
        "falls": [],
    }

    targets = {
        "jump_types": [],
        "rotations": [],
        "underrotations": [],
        "falls": [],
    }

    for batch in loader:
        x = batch["features"].float().to(device)

        y_jump_type = batch["jump_types"].long().to(device)
        y_rotations = batch["rotations"].long().to(device)
        y_underrotation = batch["underrotations"].long().to(device)
        y_fall = batch["falls"].long().to(device)

        out = model(x)

        # print(f"Eval: {y_fall = }")
        # print(f"Eval: {out['fall_logits'] = }")

        preds["jump_types"].append(out["jump_type_logits"].argmax(dim=1).cpu())
        preds["rotations"].append(out["rotations_logits"].argmax(dim=1).cpu())
        preds["underrotations"].append(out["underrotation_logits"].argmax(dim=1).cpu())
        preds["falls"].append(out["fall_logits"].argmax(dim=1).cpu())

        targets["jump_types"].append(y_jump_type.cpu())
        targets["rotations"].append(y_rotations.cpu())
        targets["underrotations"].append(y_underrotation.cpu())
        targets["falls"].append(y_fall.cpu())

    # concat
    for k in preds:
        preds[k] = torch.cat(preds[k])
        targets[k] = torch.cat(targets[k])

    return {
        "jump_type_f1": compute_f1(preds["jump_types"], targets["jump_types"], num_classes=6),
        "rotations_f1": compute_f1(preds["rotations"], targets["rotations"], num_classes=5),
        "underrotation_f1": compute_f1(preds["underrotations"], targets["underrotations"], num_classes=2),
        "fall_f1": compute_f1(preds["falls"], targets["falls"], num_classes=2),
    }