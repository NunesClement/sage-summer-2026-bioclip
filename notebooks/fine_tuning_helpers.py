"""Constrained image-encoder adaptation for the Peromyscus lesson."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import io
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


class ImageBytesDataset(Dataset):
    """Decode in-memory image bytes and apply an OpenCLIP transform."""

    def __init__(
        self,
        image_bytes: Sequence[bytes],
        targets: Sequence[int],
        transform,
    ) -> None:
        self.image_bytes = image_bytes
        self.targets = np.asarray(targets, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(io.BytesIO(self.image_bytes[index])) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.targets[index])


def configure_last_visual_block(model) -> list[str]:
    """Freeze the model except its final visual block, norm, and projection."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    visual = model.visual
    for parameter in visual.transformer.resblocks[-1].parameters():
        parameter.requires_grad_(True)
    for parameter in visual.ln_post.parameters():
        parameter.requires_grad_(True)

    projection = visual.proj
    if isinstance(projection, torch.nn.Parameter):
        projection.requires_grad_(True)
    elif isinstance(projection, torch.nn.Module):
        for parameter in projection.parameters():
            parameter.requires_grad_(True)

    return [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def parameter_summary(model) -> dict[str, float | int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": 100 * trainable / total,
    }


def trainable_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state_dict(
    model,
    state_dict: dict[str, torch.Tensor],
) -> None:
    parameters = dict(model.named_parameters())
    expected = {
        name for name, parameter in parameters.items()
        if parameter.requires_grad
    }
    if set(state_dict) != expected:
        raise ValueError("Fine-tuning checkpoint does not match trainable parameters")
    with torch.no_grad():
        for name, value in state_dict.items():
            parameters[name].copy_(
                value.to(
                    device=parameters[name].device,
                    dtype=parameters[name].dtype,
                )
            )


def save_adaptation_checkpoint(
    path: str | Path,
    *,
    manifest: dict,
    state_dict: dict[str, torch.Tensor],
    history: list[dict],
    best_epoch: int,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "manifest": manifest,
            "state_dict": state_dict,
            "history": history,
            "best_epoch": best_epoch,
        },
        destination,
    )


def load_adaptation_checkpoint(
    path: str | Path,
    *,
    expected_manifest: dict,
) -> dict | None:
    source = Path(path)
    if not source.exists():
        return None
    checkpoint = torch.load(
        source,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("manifest") != expected_manifest:
        return None
    return checkpoint


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _prototype_metrics(
    model,
    loader: DataLoader,
    prototypes: torch.Tensor,
    device: torch.device,
    logit_scale: torch.Tensor,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    examples_seen = 0
    predictions = []
    targets = []
    with torch.inference_mode():
        for images, batch_targets in loader:
            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )
            device_targets = batch_targets.to(
                device,
                non_blocking=device.type == "cuda",
            )
            with _autocast_context(device):
                features = model.encode_image(images, normalize=True)
                logits = logit_scale * features @ prototypes.T
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    device_targets,
                    reduction="sum",
                )
            total_loss += float(loss)
            examples_seen += len(batch_targets)
            predictions.append(logits.argmax(dim=1).cpu())
            targets.append(batch_targets)

    truth = torch.cat(targets).numpy()
    predicted = torch.cat(predictions).numpy()
    accuracy = float(np.mean(truth == predicted))
    class_f1 = []
    for class_index in np.unique(truth):
        true_positive = np.sum(
            (truth == class_index) & (predicted == class_index)
        )
        false_positive = np.sum(
            (truth != class_index) & (predicted == class_index)
        )
        false_negative = np.sum(
            (truth == class_index) & (predicted != class_index)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1.append(
            0.0 if denominator == 0 else 2 * true_positive / denominator
        )
    return (
        total_loss / examples_seen,
        accuracy,
        float(np.mean(class_f1)),
    )


def train_last_visual_block(
    model,
    *,
    training_images: Sequence[bytes],
    training_targets: Sequence[int],
    validation_images: Sequence[bytes],
    validation_targets: Sequence[int],
    training_transform,
    evaluation_transform,
    prototypes: np.ndarray,
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    prototype_temperature: float,
    maximum_epochs: int,
    patience: int,
    random_seed: int,
) -> dict:
    """Adapt the image encoder toward fixed text prototypes."""
    torch.manual_seed(random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_seed)

    prototype_tensor = torch.as_tensor(
        prototypes,
        dtype=torch.float32,
        device=device,
    )
    train_dataset = ImageBytesDataset(
        training_images,
        training_targets,
        training_transform,
    )
    validation_dataset = ImageBytesDataset(
        validation_images,
        validation_targets,
        evaluation_transform,
    )
    generator = torch.Generator().manual_seed(random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )
    criterion = torch.nn.CrossEntropyLoss()
    logit_scale = torch.tensor(
        1 / prototype_temperature,
        dtype=torch.float32,
        device=device,
    )

    baseline_loss, baseline_accuracy, baseline_f1 = _prototype_metrics(
        model,
        validation_loader,
        prototype_tensor,
        device,
        logit_scale,
    )
    best_state = trainable_state_dict(model)
    best_epoch = 0
    best_validation_f1 = baseline_f1
    best_validation_loss = baseline_loss
    lowest_validation_loss = baseline_loss
    epochs_without_improvement = 0
    history = [{
        "epoch": 0,
        "training_loss": float("nan"),
        "validation_loss": baseline_loss,
        "validation_accuracy": baseline_accuracy,
        "validation_macro_f1": baseline_f1,
    }]
    print(
        "Epoch 00: "
        f"validation loss={baseline_loss:.3f}, "
        f"validation macro-F1={baseline_f1:.3f} "
        "(off-the-shelf checkpoint)"
    )

    for epoch in range(1, maximum_epochs + 1):
        model.eval()
        model.visual.transformer.resblocks[-1].train()
        running_loss = 0.0
        examples_seen = 0

        for images, targets in train_loader:
            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )
            targets = targets.to(
                device,
                non_blocking=device.type == "cuda",
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device):
                features = model.encode_image(images, normalize=True)
                logits = logit_scale * features @ prototype_tensor.T
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_examples = len(targets)
            running_loss += float(loss.detach()) * batch_examples
            examples_seen += batch_examples

        (
            validation_loss,
            validation_accuracy,
            validation_f1,
        ) = _prototype_metrics(
            model,
            validation_loader,
            prototype_tensor,
            device,
            logit_scale,
        )
        epoch_row = {
            "epoch": epoch,
            "training_loss": running_loss / examples_seen,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_macro_f1": validation_f1,
        }
        history.append(epoch_row)
        print(
            f"Epoch {epoch:02d}: "
            f"loss={epoch_row['training_loss']:.3f}, "
            f"validation loss={validation_loss:.3f}, "
            f"validation macro-F1={validation_f1:.3f}"
        )

        better_checkpoint = (
            validation_f1 > best_validation_f1
            or (
                np.isclose(validation_f1, best_validation_f1)
                and validation_loss < best_validation_loss
            )
        )
        if better_checkpoint:
            best_validation_f1 = validation_f1
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(trainable_state_dict(model))

        if validation_loss < lowest_validation_loss:
            lowest_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Stopped after {epoch} epochs; validation loss "
                    f"did not improve for {patience} epochs."
                )
                break

    load_trainable_state_dict(model, best_state)
    model.eval()
    return {
        "state_dict": best_state,
        "history": history,
        "best_epoch": best_epoch,
    }
