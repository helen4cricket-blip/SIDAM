from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = WORKSPACE / "data" / "earthquake_damage" / "Earthquake"
DEFAULT_OUTPUT_DIR = WORKSPACE / "models"
CHECKPOINT_NAME = "damage_convnext_tiny.pth"
SEED = 42


def split_indices(
    targets: list[int], class_count: int, seed: int
) -> tuple[list[int], list[int], list[int]]:
    generator = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []

    for class_index in range(class_count):
        indices = [i for i, target in enumerate(targets) if target == class_index]
        generator.shuffle(indices)
        test_size = max(1, round(len(indices) * 0.15))
        validation_size = max(1, round(len(indices) * 0.15))
        test_indices.extend(indices[:test_size])
        validation_indices.extend(indices[test_size : test_size + validation_size])
        train_indices.extend(indices[test_size + validation_size :])

    generator.shuffle(train_indices)
    generator.shuffle(validation_indices)
    generator.shuffle(test_indices)
    return train_indices, validation_indices, test_indices


def make_model(class_count: int, pretrained: bool) -> nn.Module:
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    if training:
        if any(parameter.requires_grad for parameter in model.features.parameters()):
            model.features[:-1].eval()
        else:
            model.features.eval()

    loss_total = 0.0
    confusion = torch.zeros((2, 2), dtype=torch.int64)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        loss_total += loss.item() * labels.size(0)
        encoded = labels.cpu() * 2 + predictions.cpu()
        confusion += torch.bincount(encoded, minlength=4).reshape(2, 2)

    return loss_total / len(loader.dataset), confusion


def print_metrics(title: str, loss: float, confusion: torch.Tensor, class_names: list[str]) -> None:
    total = int(confusion.sum())
    accuracy = int(confusion.diag().sum()) / total if total else 0.0
    recalls = []
    print(f"\n{title}: loss={loss:.4f}, accuracy={accuracy:.3f}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion.tolist())

    for index, name in enumerate(class_names):
        actual = int(confusion[index].sum())
        correct = int(confusion[index, index])
        recall = correct / actual if actual else 0.0
        recalls.append(recall)
        print(f"  {name}: recall={recall:.3f} ({correct}/{actual})")

    print(f"  balanced accuracy={sum(recalls) / len(recalls):.3f}")


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, weights.transforms()


def train(args: argparse.Namespace) -> None:
    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {data_dir}")

    probe = datasets.ImageFolder(data_dir)
    expected_classes = {"Damaged Building", "Undamaged Building"}
    if set(probe.classes) != expected_classes:
        raise ValueError(
            f"Expected class folders {sorted(expected_classes)}, found {probe.classes} "
            f"under {data_dir}"
        )

    class_names = probe.classes
    class_counts = Counter(probe.targets)
    print(f"Classes: {probe.class_to_idx}")
    print(
        "Images: "
        + ", ".join(
            f"{name}={class_counts[index]}" for index, name in enumerate(class_names)
        )
    )

    train_indices, validation_indices, test_indices = split_indices(
        probe.targets, len(class_names), SEED
    )
    train_transform, evaluation_transform = make_transforms()
    train_data = datasets.ImageFolder(data_dir, transform=train_transform)
    evaluation_data = datasets.ImageFolder(data_dir, transform=evaluation_transform)

    train_loader = DataLoader(
        Subset(train_data, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(evaluation_data, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        Subset(evaluation_data, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    model = make_model(len(class_names), pretrained=True).to(device)
    train_class_counts = torch.bincount(
        torch.tensor([probe.targets[i] for i in train_indices]),
        minlength=len(class_names),
    ).float()
    class_weights = (train_class_counts.sum() / (len(class_names) * train_class_counts)).to(
        device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_NAME
    best_validation_loss = float("inf")

    phases = [
        ("classifier", args.head_epochs, 1e-3),
        ("fine-tune final stage", args.finetune_epochs, 1e-5),
    ]
    completed_epoch = 0

    for phase_name, epoch_count, learning_rate in phases:
        if epoch_count <= 0:
            continue
        for parameter in model.features.parameters():
            parameter.requires_grad = phase_name != "classifier"
        if phase_name == "fine-tune final stage":
            for parameter in model.features[:-1].parameters():
                parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=0.01,
        )

        print(f"\nPhase: {phase_name}")
        for phase_epoch in range(epoch_count):
            completed_epoch += 1
            start = time.perf_counter()
            train_loss, _ = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
            )
            validation_loss, validation_confusion = run_epoch(
                model, validation_loader, criterion, device
            )
            elapsed = time.perf_counter() - start
            print(
                f"Epoch {completed_epoch}: train_loss={train_loss:.4f}, "
                f"validation_loss={validation_loss:.4f}, "
                f"validation_accuracy="
                f"{validation_confusion.diag().sum().item() / len(validation_indices):.3f}, "
                f"time={elapsed:.1f}s"
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(
                    {
                        "architecture": "convnext_tiny",
                        "class_names": class_names,
                        "model_state_dict": model.state_dict(),
                        "validation_loss": validation_loss,
                        "seed": SEED,
                    },
                    checkpoint_path,
                )

    if not checkpoint_path.is_file():
        raise RuntimeError("Training completed without saving a model checkpoint.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_confusion = run_epoch(model, test_loader, criterion, device)
    print_metrics("Held-out test results", test_loss, test_confusion, class_names)

    metrics_path = output_dir / "training_metrics.json"
    metrics = {
        "architecture": "ConvNeXt-Tiny",
        "device": str(device),
        "class_to_index": probe.class_to_idx,
        "image_counts": {
            name: class_counts[index] for index, name in enumerate(class_names)
        },
        "split_counts": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "best_validation_loss": checkpoint["validation_loss"],
        "test_loss": test_loss,
        "test_accuracy": float(test_confusion.diag().sum().item() / len(test_indices)),
        "test_confusion_matrix": test_confusion.tolist(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved model: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")


def predict(image_path: Path, checkpoint_path: Path) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}. Train the model first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]
    model = make_model(len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = ConvNeXt_Tiny_Weights.DEFAULT.transforms()
    with torch.inference_mode():
        logits = model(transform(image).unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu()

    predicted_index = int(probabilities.argmax())
    print(f"Image: {image_path}")
    print(f"Prediction: {class_names[predicted_index]}")
    print(f"Confidence: {probabilities[predicted_index].item():.1%}")
    for index, name in enumerate(class_names):
        print(f"  {name}: {probabilities[index].item():.1%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or run a ConvNeXt-Tiny damaged/undamaged image classifier."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument(
        "--predict",
        type=Path,
        help="Classify one image using the saved checkpoint instead of training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / CHECKPOINT_NAME,
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.head_epochs < 0 or args.finetune_epochs < 0:
        parser.error("Epoch counts cannot be negative")
    if args.predict is None and args.head_epochs + args.finetune_epochs == 0:
        parser.error("At least one training epoch is required")
    return args


def main() -> None:
    args = parse_args()
    if args.predict is not None:
        predict(args.predict, args.checkpoint)
    else:
        train(args)


if __name__ == "__main__":
    main()
