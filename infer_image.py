import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_model():
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model.load_state_dict(torch.load("efficientnet_ffpp_best.pth",
                                     map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict(path):
    img = Image.open(path).convert("RGB")
    x = tfms(img).unsqueeze(0).to(DEVICE)

    model = get_model()
    with torch.no_grad():
        logit = model(x)
        prob_fake = torch.sigmoid(logit).item()

    label = "FAKE" if prob_fake >= 0.5 else "REAL"
    print(f"{Path(path).name}: {label} (fake prob = {prob_fake:.3f})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer_image.py path/to/image.jpg")
        sys.exit(1)
    predict(sys.argv[1])