import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

# Load CASIA2-trained model
def load_objects_model(model_path="S:/Python/TrustMediaBackend/DeepfakeDetector/efficientnet_casia_best.pth"):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

_objects_model = None

def get_objects_model():
    global _objects_model
    if _objects_model is None:
        _objects_model = load_objects_model()
    return _objects_model

_obj_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def get_object_fake_score(image_path: str) -> float:
    img = Image.open(image_path).convert("RGB")
    x = _obj_tfms(img).unsqueeze(0)

    model = get_objects_model()
    with torch.no_grad():
        logit = model(x)
        fake_score = torch.sigmoid(logit)[0].item()
    return float(fake_score)
