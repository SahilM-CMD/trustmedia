import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def load_model(model_path="efficientnet_ffpp_best.pth"):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 1)
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

if __name__ == "__main__":
    model = load_model("S:/Python/TrustMediaBackend/DeepfakeDetector/efficientnet_ffpp_best.pth")

    dummy = torch.randn(1, 3, 224, 224)
    scripted = torch.jit.trace(model, dummy)
    scripted.save("efficientnet_ffpp_best_mobile.pt")
    print("Saved TorchScript model: efficientnet_ffpp_best_mobile.pt")
