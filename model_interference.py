# DeepfakeDetector/model_inference.py

import io
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _load_model():
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    state_path = Path(__file__).parent / "efficientnet_ffpp_best.pth"
    model.load_state_dict(torch.load(state_path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

_model = _load_model()

_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_image_path(path: str) -> dict:
    """Takes an image file path, returns dict with label and prob_fake."""
    img = Image.open(path).convert("RGB")
    x = _tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = _model(x)
        prob_fake = torch.sigmoid(logit).item()

    label = "FAKE" if prob_fake >= 0.5 else "REAL"
    return {
        "label": label,
        "prob_fake": float(prob_fake),
    }
