import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

DATA_DIR = "/kaggle/input/chest-xray-pneumonia/chest_xray/test"
CHECKPOINT = "/content/lab1_outputs/resnet18_best.pt"
OUTPUT_PATH = "/content/lab1_outputs/figure4_resnet18_final_test.png"

MODEL_NAME = "resnet18"
BATCH_SIZE = 32

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


class ChestXrayDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        extensions = {".jpg", ".jpeg", ".png"}

        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = self.root / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"找不到資料夾：{class_dir}\n"
                    "請修改 DATA_DIR，使其底下包含 NORMAL 和 PNEUMONIA。"
                )

            for image_path in sorted(class_dir.rglob("*")):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in extensions
                ):
                    self.samples.append((image_path, label))

        print(f"找到 {len(self.samples)} 張測試影像")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

        return image, label


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_dataset = ChestXrayDataset(
    DATA_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)


if MODEL_NAME == "resnet18":
    model = models.resnet18(weights=None)
elif MODEL_NAME == "resnet50":
    model = models.resnet50(weights=None)
else:
    raise ValueError("MODEL_NAME 必須是 resnet18 或 resnet50")

model.fc = nn.Linear(model.fc.in_features, 2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

if isinstance(checkpoint, nn.Module):
    state_dict = checkpoint.state_dict()
elif "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint

clean_state_dict = {}

for key, value in state_dict.items():
    key = key.removeprefix("module.")
    key = key.removeprefix("_orig_mod.")
    clean_state_dict[key] = value

model.load_state_dict(clean_state_dict, strict=True)
model = model.to(device)
model.eval()

print(f"使用裝置：{device}")
print(f"載入 checkpoint：{CHECKPOINT}")

y_true = []
y_pred = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1).cpu()

        y_true.extend(labels.numpy())
        y_pred.extend(predictions.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)


accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, pos_label=1)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

print("\n===== Final Test Result =====")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F1-score: {f1:.4f}")
print("\nConfusion matrix:")
print(cm)


os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

plt.figure(figsize=(7, 5.5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    cbar=True,
    annot_kws={"size": 14}
)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title(
    f"Final Test Confusion Matrix: ResNet18\n"
    f"Accuracy = {accuracy * 100:.2f}%, F1 = {f1:.4f}"
)

plt.tight_layout()
plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print(f"\nFigure 4 已儲存到：{OUTPUT_PATH}")
