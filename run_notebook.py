"""
run_notebook.py
===============
Equivalent of all cells in the Phase 2 notebook, runnable as a plain Python
script.  Produces the same outputs the notebook would (training curves,
confusion matrix, labeled unseen-image predictions, saved model).
"""

import copy
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for script mode
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ──────────────────────────── 1. Paths ─────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "validation"
TEST_DIR = DATA_ROOT / "test"
UNSEEN_DIR = PROJECT_ROOT / "data" / "raw" / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Train dir:", TRAIN_DIR)
print("Validation dir:", VAL_DIR)
print("Test dir:", TEST_DIR)
print("Unseen dir:", UNSEEN_DIR)

# ──────────────────────────── 2. Validate folders ──────────────────────────

def list_label_folders(split_dir: Path):
    if not split_dir.exists():
        return []
    return sorted([item.name for item in split_dir.iterdir() if item.is_dir()])

for split_name, split_dir in [("train", TRAIN_DIR), ("validation", VAL_DIR), ("test", TEST_DIR)]:
    labels = list_label_folders(split_dir)
    print(f"{split_name}: exists={split_dir.exists()} | labels={labels}")

assert TRAIN_DIR.exists(), "Training directory is missing."
assert VAL_DIR.exists(), "Validation directory is missing."
assert TEST_DIR.exists(), "Test directory is missing."
assert list_label_folders(TRAIN_DIR), "No class folders found in the training directory."

# ──────────────────────────── 3. Setup ─────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ──────────────────────────── 4. Transforms ────────────────────────────────

IMAGE_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

print({"batch_size": BATCH_SIZE, "epochs": EPOCHS, "image_size": IMAGE_SIZE})

# ──────────────────────────── 5. Datasets & Loaders ────────────────────────

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

class_names = train_dataset.classes
num_classes = len(class_names)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Classes:", class_names)
print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

# ──────────────────────────── 6. Class distribution ────────────────────────

train_counts = Counter([class_names[label] for _, label in train_dataset.samples])
val_counts = Counter([class_names[label] for _, label in val_dataset.samples])
test_counts = Counter([class_names[label] for _, label in test_dataset.samples])

print("Train distribution:", dict(train_counts))
print("Validation distribution:", dict(val_counts))
print("Test distribution:", dict(test_counts))

# ──────────────────────────── 7. Visualize augmented images ────────────────

def denormalize(image_tensor):
    image = image_tensor.clone().cpu().numpy().transpose(1, 2, 0)
    image = image * np.array(imagenet_std) + np.array(imagenet_mean)
    return np.clip(image, 0, 1)

images, labels = next(iter(train_loader))
fig, axes = plt.subplots(1, min(4, len(images)), figsize=(16, 4))
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])
for axis, image, label in zip(axes, images[:4], labels[:4]):
    axis.imshow(denormalize(image))
    axis.set_title(class_names[label.item()])
    axis.axis("off")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "augmented_samples.png", dpi=150)
plt.close(fig)
print("Saved augmented samples plot.")

# ──────────────────────────── 8. Build model ───────────────────────────────

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

print("Model:", model.__class__.__name__)

# ──────────────────────────── 9. Train ─────────────────────────────────────

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
}

best_model_state = copy.deepcopy(model.state_dict())
best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch_images, batch_labels in train_loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_images.size(0)
        predictions = outputs.argmax(dim=1)
        running_correct += (predictions == batch_labels).sum().item()
        running_total += batch_labels.size(0)

    train_loss = running_loss / running_total
    train_accuracy = running_correct / running_total

    model.eval()
    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            val_loss_total += loss.item() * batch_images.size(0)
            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == batch_labels).sum().item()
            val_total += batch_labels.size(0)

    val_loss = val_loss_total / val_total
    val_accuracy = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = copy.deepcopy(model.state_dict())

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
        f"val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f}"
    )

model.load_state_dict(best_model_state)

# ──────────────────────────── 10. Training curves ──────────────────────────

epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(epochs_range, history["train_loss"], label="Train Loss")
axes[0].plot(epochs_range, history["val_loss"], label="Validation Loss")
axes[0].set_title("Loss Over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(epochs_range, history["train_accuracy"], label="Train Accuracy")
axes[1].plot(epochs_range, history["val_accuracy"], label="Validation Accuracy")
axes[1].set_title("Accuracy Over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
plt.close(fig)
print("Saved training curves.")

# ──────────────────────────── 11. Test evaluation ──────────────────────────

model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for batch_images, batch_labels in test_loader:
        batch_images = batch_images.to(device)
        outputs = model(batch_images)
        predictions = outputs.argmax(dim=1).cpu().numpy()
        all_predictions.extend(predictions.tolist())
        all_labels.extend(batch_labels.numpy().tolist())

test_accuracy = accuracy_score(all_labels, all_predictions)
report = classification_report(all_labels, all_predictions, target_names=class_names, digits=4)
cm = confusion_matrix(all_labels, all_predictions)

print(f"\nTest accuracy: {test_accuracy:.4f}")
print(report)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig)
print("Saved confusion matrix.")

metrics_path = OUTPUT_DIR / "phase2_metrics.json"
metrics_path.write_text(json.dumps({"test_accuracy": test_accuracy, "classes": class_names}, indent=2))
print("Saved metrics to", metrics_path)

# ──────────────────────────── 12. Classify unseen data ─────────────────────

output_prediction_dir = PROJECT_ROOT / "outputs" / "predictions"
output_prediction_dir.mkdir(parents=True, exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
unseen_files = [path for path in sorted(UNSEEN_DIR.rglob("*")) if path.suffix.lower() in image_extensions]
print("\nUnseen images found:", len(unseen_files))

def predict_image(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)
    return image, class_names[predicted_index.item()], confidence.item()

preview_count = min(6, len(unseen_files))
if preview_count == 0:
    print("No unseen images available. Update UNSEEN_DIR if needed.")
else:
    fig, axes = plt.subplots(
        math.ceil(preview_count / 3),
        min(3, preview_count),
        figsize=(15, 4 * math.ceil(preview_count / 3)),
    )
    axes = np.array(axes).reshape(-1)

    for axis, image_path in zip(axes, unseen_files[:preview_count]):
        image, predicted_label, confidence = predict_image(image_path)
        draw = ImageDraw.Draw(image)
        label_text = f"{predicted_label} ({confidence * 100:.1f}%)"

        # Try to use a larger, more visible font
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except OSError:
            font = ImageFont.load_default()

        # Draw label background and text
        bbox = draw.textbbox((15, 15), label_text, font=font)
        draw.rectangle(
            [(bbox[0] - 5, bbox[1] - 5), (bbox[2] + 5, bbox[3] + 5)],
            fill="black",
        )
        draw.text((15, 15), label_text, fill="lime", font=font)

        save_path = output_prediction_dir / f"labeled_{image_path.name}"
        image.save(save_path)

        axis.imshow(image)
        axis.set_title(image_path.name)
        axis.axis("off")

    for axis in axes[preview_count:]:
        axis.axis("off")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "unseen_predictions.png", dpi=150)
    plt.close(fig)
    print("Saved unseen predictions plot.")
    print("Saved labeled predictions to", output_prediction_dir)

    # Also classify ALL unseen images (not just preview)
    for image_path in unseen_files:
        image, predicted_label, confidence = predict_image(image_path)
        draw = ImageDraw.Draw(image)
        label_text = f"{predicted_label} ({confidence * 100:.1f}%)"
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((15, 15), label_text, font=font)
        draw.rectangle(
            [(bbox[0] - 5, bbox[1] - 5), (bbox[2] + 5, bbox[3] + 5)],
            fill="black",
        )
        draw.text((15, 15), label_text, fill="lime", font=font)
        save_path = output_prediction_dir / f"labeled_{image_path.name}"
        image.save(save_path)

# ──────────────────────────── 13. Save model ───────────────────────────────

model_output_path = PROJECT_ROOT / "data" / "models" / "phase2_resnet18_classifier.pt"
model_output_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
    },
    model_output_path,
)
print("Saved model to", model_output_path)
print("\n✅ Phase 2 complete!")
