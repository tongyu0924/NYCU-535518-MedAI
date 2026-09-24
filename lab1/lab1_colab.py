"""Lab 1: chest X-ray pneumonia classification. One-file Google Colab program.

In Colab: upload this file, choose Runtime > Change runtime type > GPU,
then run:  !python lab1_colab.py

Public Kaggle data is downloaded automatically. To use an existing dataset or
change training settings, edit the settings block below and run again.

Outputs go to /content/lab1_outputs in Colab (./lab1_outputs elsewhere).
"""

import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_NAMES = ("NORMAL", "PNEUMONIA")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ===== Settings: edit these values if needed, then run the file directly. =====
DATA_DIR = None  # Example: "/content/chest_xray"; None downloads the public dataset
OUTPUT_DIR = "/content/lab1_outputs" if Path("/content").exists() else "lab1_outputs"
EPOCHS = 10
BATCH_SIZE = 32
IMAGE_SIZE = 224
VAL_FRACTION = 0.15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
WORKERS = 2
PRETRAINED = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_data_root(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    candidates = [path] + [p for p in path.rglob("train") if p.is_dir()]
    for p in candidates:
        root = p.parent if p.name == "train" else p
        if all((root / split / cls).is_dir()
               for split in ("train", "test") for cls in CLASS_NAMES):
            return root
    raise FileNotFoundError(
        f"Could not find train/NORMAL, train/PNEUMONIA, test/NORMAL, "
        f"test/PNEUMONIA under {path}"
    )


def get_data_root(data_dir):
    if data_dir:
        return find_data_root(data_dir)
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub to download the public Kaggle dataset...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])
        import kagglehub
    try:
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    except Exception as exc:
        raise RuntimeError(
            "Automatic dataset download failed. Download the Kaggle chest X-ray "
            "dataset and rerun with --data_dir /path/to/chest_xray."
        ) from exc
    return find_data_root(path)


def collect_samples(root, split):
    samples = []
    for label, cls in enumerate(CLASS_NAMES):
        samples += [(p, label) for p in sorted((root / split / cls).rglob("*"))
                    if p.is_file() and p.suffix.lower() in EXTENSIONS]
    if not samples or {label for _, label in samples} != {0, 1}:
        raise ValueError(f"Both classes need images in {root / split}")
    return samples


class ChestXrayDataset(Dataset):
    """Custom PyTorch dataset: each image path and binary label is explicit."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return image, label


def make_loaders(root, image_size, batch_size, val_fraction, seed, workers):
    from sklearn.model_selection import train_test_split

    all_train = collect_samples(root, "train")
    train, val = train_test_split(
        all_train, test_size=val_fraction, random_state=seed,
        stratify=[label for _, label in all_train]
    )
    test = collect_samples(root, "test")
    normalizer = transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(), normalizer,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), normalizer,
    ])
    loader_args = dict(batch_size=batch_size, num_workers=workers,
                       pin_memory=torch.cuda.is_available())
    loaders = {
        "train": DataLoader(ChestXrayDataset(train, train_tf), shuffle=True,
                            **loader_args),
        "val": DataLoader(ChestXrayDataset(val, eval_tf), shuffle=False,
                          **loader_args),
        "test": DataLoader(ChestXrayDataset(test, eval_tf), shuffle=False,
                           **loader_args),
    }
    print(f"Dataset: {root} | train={len(train)}, val={len(val)}, test={len(test)}")
    return loaders, train


def make_model(name, pretrained):
    constructor = getattr(models, name)
    weights = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
    }[name] if pretrained else None
    try:
        model = constructor(weights=weights)
    except Exception as exc:
        if not pretrained:
            raise
        print(f"WARNING: {name} pretrained weights unavailable ({exc}); "
              "starting with random weights.")
        model = constructor(weights=None)
    # Always replace ImageNet's final layer with a freshly initialized binary head.
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def scores(confusion):
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    total = int(confusion.sum())
    return {
        "accuracy": 100 * (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def one_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    confusion = np.zeros((2, 2), dtype=np.int64)
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            actual = labels.detach().cpu().numpy()
            predicted = outputs.argmax(dim=1).detach().cpu().numpy()
            np.add.at(confusion, (actual, predicted), 1)
    return {"loss": total_loss / len(loader.dataset), **scores(confusion)}, confusion


def plot_curves(history, output):
    epochs = range(1, len(next(iter(history.values()))) + 1)
    for metric, unit in (("accuracy", "%"), ("f1", "")):
        fig, ax = plt.subplots(figsize=(9, 5))
        for model_name, rows in history.items():
            for split, style in (("train", "-"), ("val", "--"), ("test", ":")):
                ax.plot(epochs, [row[f"{split}_{metric}"] for row in rows],
                        style, marker="o", markersize=3,
                        label=f"{model_name} {split}")
        ax.set(xlabel="Epoch", ylabel=f"{metric.title()} {unit}",
               title=f"Chest X-ray {metric.title()} by epoch")
        ax.set_xticks(list(epochs))
        ax.grid(alpha=0.3)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(output / f"{metric}_curves.png", dpi=160)
        plt.close(fig)


def plot_confusion(confusion, name, output):
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=(0, 1), yticks=(0, 1), xticklabels=CLASS_NAMES,
           yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="True",
           title=f"{name} test confusion matrix")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(confusion[row, col]), ha="center", va="center",
                    color="white" if confusion[row, col] > confusion.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(output / f"{name}_confusion_matrix.png", dpi=160)
    plt.close(fig)


def run():
    if EPOCHS < 1 or BATCH_SIZE < 1 or not 0 < VAL_FRACTION < 0.5:
        raise ValueError("Use epochs/batch_size >= 1 and 0 < val_fraction < 0.5")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} (GPU strongly recommended)")
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    root = get_data_root(DATA_DIR)
    loaders, train_samples = make_loaders(root, IMAGE_SIZE, BATCH_SIZE,
                                         VAL_FRACTION, SEED, WORKERS)
    counts = np.bincount([label for _, label in train_samples], minlength=2)
    class_weights = torch.tensor(len(train_samples) / (2 * counts),
                                 dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    all_history = {}
    summary = {}

    for name in ("resnet18", "resnet50"):
        print(f"\n=== {name} ===")
        model = make_model(name, PRETRAINED).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                      weight_decay=WEIGHT_DECAY)
        history = []
        best_f1 = -1.0
        best_acc = -1.0
        weights_path = output / f"{name}_best.pt"
        for epoch in range(1, EPOCHS + 1):
            train_stats, _ = one_epoch(model, loaders["train"], criterion,
                                       device, optimizer)
            val_stats, _ = one_epoch(model, loaders["val"], criterion, device)
            # The assignment asks for test accuracy/F1 curves at every epoch.
            # Test results are recorded only; checkpoint selection uses validation.
            test_stats, _ = one_epoch(model, loaders["test"], criterion, device)
            row = {"model": name, "epoch": epoch}
            for split, stats in (("train", train_stats), ("val", val_stats),
                                 ("test", test_stats)):
                row.update({f"{split}_{key}": value for key, value in stats.items()})
            history.append(row)
            print(f"Epoch {epoch:02d}/{EPOCHS}: "
                  f"train acc={train_stats['accuracy']:.2f}% "
                  f"val acc={val_stats['accuracy']:.2f}% "
                  f"val F1={val_stats['f1']:.4f} "
                  f"test acc={test_stats['accuracy']:.2f}% "
                  f"test F1={test_stats['f1']:.4f}")
            if (val_stats["f1"], val_stats["accuracy"]) > (best_f1, best_acc):
                best_f1, best_acc = val_stats["f1"], val_stats["accuracy"]
                torch.save(model.state_dict(), weights_path)
                best_epoch = epoch

        model.load_state_dict(torch.load(weights_path, map_location=device,
                                         weights_only=True))
        final_stats, confusion = one_epoch(model, loaders["test"], criterion, device)
        plot_confusion(confusion, name, output)
        summary[name] = {"best_validation_epoch": best_epoch,
                         "final_test": final_stats,
                         "confusion_matrix": confusion.tolist(),
                         "weights_file": weights_path.name}
        all_history[name] = history
        print(f"Best validation epoch={best_epoch} | final test: "
              f"accuracy={final_stats['accuracy']:.2f}%, F1={final_stats['f1']:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_curves(all_history, output)
    with (output / "epoch_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_history["resnet18"][0]))
        writer.writeheader()
        for rows in all_history.values():
            writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFinished. All figures, weights and metrics: {output.resolve()}")

    # Package the entire results folder, including both best model weights.
    zip_path = shutil.make_archive(str(output.resolve()), "zip",
                                   root_dir=output.parent.resolve(),
                                   base_dir=output.name)
    print(f"Download archive ready: {zip_path}")
    try:
        from google.colab import files
    except ImportError:
        pass  # Outside Colab, the ZIP is available at the printed path.
    else:
        files.download(zip_path)


run()
