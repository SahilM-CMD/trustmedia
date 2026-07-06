import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("../data/ffpp_processed")
BATCH_SIZE = 32

def get_test_loader():
    test_dir = DATA_ROOT / "test"

    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_ds = datasets.ImageFolder(str(test_dir), transform=tfms)
    print("Test class_to_idx:", test_ds.class_to_idx)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)
    return test_loader

def get_model():
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model.load_state_dict(torch.load("efficientnet_ffpp_best.pth",
                                     map_location=DEVICE))
    return model.to(DEVICE)

def evaluate():
    loader = get_test_loader()
    model = get_model()
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

    print(f"Test accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    evaluate()
