import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============ PATHS ============
DATA_ROOT = r"S:\Python\TrustMediaBackend\data"
CASIA2_DIR = os.path.join(DATA_ROOT, "CASIA2")
FFPP_DIR = os.path.join(DATA_ROOT, "ffpp_processed")
TEST_MATERIAL = r"S:\Python\TrustMediaBackend\DeepfakeDetector\Testing_Material"
MODEL_SAVE_PATH = r"S:\Python\TrustMediaBackend\DeepfakeDetector\efficientnet_ffpp_best_finetuned.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============ DATASET CLASS ============
class FakeFaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy data on error
            dummy_img = Image.new("RGB", (224, 224))
            if self.transform:
                dummy_img = self.transform(dummy_img)
            return dummy_img, torch.tensor(label, dtype=torch.float32)


# ============ LOAD IMAGES ============
def load_images_from_path(folder_path, label, max_images=500):
    """Load image paths from folder"""
    image_paths = []
    labels = []
    count = 0

    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} not found")
        return image_paths, labels

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                if count >= max_images:
                    return image_paths, labels

                file_path = os.path.join(root, file)
                image_paths.append(file_path)
                labels.append(label)
                count += 1

                if count % 50 == 0:
                    print(f"  Loaded {count} images from {os.path.basename(folder_path)}")

    return image_paths, labels


print("\n=== LOADING TRAINING DATA ===")
print("Loading CASIA2 (fake faces)...")
casia_paths, casia_labels = load_images_from_path(CASIA2_DIR, label=1, max_images=300)

print("Loading FFPP (fake videos/frames)...")
ffpp_paths, ffpp_labels = load_images_from_path(FFPP_DIR, label=1, max_images=300)

fake_paths = casia_paths + ffpp_paths
fake_labels = casia_labels + ffpp_labels

print(f"Loading real images from Testing_Material...")
real_paths, real_labels = load_images_from_path(TEST_MATERIAL, label=0, max_images=150)

all_paths = real_paths + fake_paths
all_labels = real_labels + fake_labels

if len(all_paths) == 0:
    print("ERROR: No images loaded!")
    exit(1)

print(f"\n=== DATASET SUMMARY ===")
print(f"Total images: {len(all_paths)}")
print(f"Real images: {sum(1 for l in all_labels if l == 0)}")
print(f"Fake images: {sum(1 for l in all_labels if l == 1)}")

# ============ SPLIT DATA ============
X_train, X_val, y_train, y_val = train_test_split(
    all_paths, all_labels,
    test_size=0.2,
    random_state=42,
    stratify=all_labels
)

print(f"\nTraining set: {len(X_train)} images")
print(f"Validation set: {len(X_val)} images")

# ============ TRANSFORMS ============
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============ CREATE DATALOADERS ============
train_dataset = FakeFaceDataset(X_train, y_train, transform=train_transform)
val_dataset = FakeFaceDataset(X_val, y_val, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# ============ BUILD MODEL ============
print("\n=== BUILDING MODEL ===")
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)

# Replace classifier for binary classification
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 1)  # 1 logit for binary classification
)

model.to(device)

# ============ TRAINING ============
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

epochs = 30
best_val_auc = 0
patience = 5
patience_counter = 0

train_losses = []
val_losses = []
train_accs = []
val_accs = []

print("\n=== TRAINING MODEL ===")
for epoch in range(epochs):
    # Train
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Calculate accuracy
        preds = (torch.sigmoid(logits) > 0.5).long()
        train_correct += (preds == labels.long()).sum().item()
        train_total += labels.size(0)

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validate
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images).squeeze()
            loss = criterion(logits, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            val_correct += (preds == labels.long()).sum().item()
            val_total += labels.size(0)

            val_probs.extend(probs.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # Calculate AUC
    from sklearn.metrics import roc_auc_score

    try:
        val_auc = roc_auc_score(val_targets, val_probs)
    except:
        val_auc = 0.5

    print(
        f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✓ Model saved! AUC: {val_auc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    scheduler.step(val_auc)

# ============ PLOT RESULTS ============
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig(r"S:\Python\TrustMediaBackend\DeepfakeDetector\pytorch_training_history.png", dpi=100)
print("\n✓ Training history saved to pytorch_training_history.png")
plt.show()

print(f"\n✓ Fine-tuned model saved to {MODEL_SAVE_PATH}")
