import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import argparse
import cv2
import numpy as np
import imageio.v3 as iio
from objects_classifier import get_object_fake_score

# -------------------------------------------------
# Global device + shared video transform
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

video_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------------------------
# Load your trained model (FF++ best weights)
# -------------------------------------------------
def load_model(model_path="S:/Python/TrustMediaBackend/DeepfakeDetector/efficientnet_ffpp_best.pth"):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    # 1 output logit (fake probability after sigmoid)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 1)
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()
    return model

# Preprocess and classify image (CLI use)
def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(input_tensor)            # shape [1, 1]
        prob_fake = torch.sigmoid(logit)[0].item()

    label = "FAKE" if prob_fake >= 0.5 else "REAL"
    print(f"\n🧠 Prediction: {label}")
    print(f"Fake prob: {prob_fake:.3f}")

# ---------- TrustMedia helper functions ----------

# get fake probability only (0..1)
def get_fake_score(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(input_tensor)            # [1, 1]
        fake_score = torch.sigmoid(logit)[0].item()
    return fake_score

def interpret_fake_score(fake_score: float) -> str:
    if fake_score >= 0.85:
        return "HIGH_SUSPICION"
    elif fake_score >= 0.6:
        return "MEDIUM_SUSPICION"
    else:
        return "LOW_SUSPICION"

# main function for your backend
_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

def check_image(path: str):
    if has_face(path):
        # Use face model (FF++)
        model = get_model()
        fake_score = get_fake_score(path, model)
        source = "face"
    else:
        # Use object model (CASIA2)
        fake_score = get_object_fake_score(path)
        source = "object"

    level = interpret_fake_score(fake_score)
    return {
        "type": source,           # "face" or "object"
        "fake_score": float(fake_score),
        "suspicion_level": level,
    }

# ---------- Video helper functions: use imageio, aggressive ----------

BATCH_SIZE = 16
FRAME_STRIDE = 3   # sample every 3rd frame

def get_video_fake_score(video_path, model) -> float:
    """
    Read video frames with imageio (ffmpeg), sample frames,
    run model, return MAX fake score.
    """
    frame_tensors = []
    scores = []
    frame_idx = 0

    try:
        for frame in iio.imiter(video_path):   # RGB frames
            if frame_idx % FRAME_STRIDE != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            pil_img = Image.fromarray(frame)
            tensor = video_transform(pil_img)
            frame_tensors.append(tensor)

            if len(frame_tensors) == BATCH_SIZE:
                batch = torch.stack(frame_tensors, dim=0).to(device)
                with torch.no_grad():
                    logits = model(batch).view(-1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                scores.extend(probs.tolist())
                frame_tensors = []
    except Exception as e:
        print("imageio error while reading video:", e)
        return 0.5

    if frame_tensors:
        batch = torch.stack(frame_tensors, dim=0).to(device)
        with torch.no_grad():
            logits = model(batch).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        scores.extend(probs.tolist())

    if not scores:
        print("NO SCORES FOUND - returning 0.5")
        return 0.5

    fake_score = float(np.max(scores))
    print(f"Video scores count: {len(scores)}, max={fake_score}")
    return fake_score

def check_video(path: str):
    model = get_model()
    fake_score = get_video_fake_score(path, model)
    level = interpret_fake_score(fake_score)
    return {
        "type": "video",
        "fake_score": float(fake_score),
        "suspicion_level": level,
    }

FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def has_face(path: str) -> bool:
    img = cv2.imread(path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

# ---------- CLI entry (image or video) ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to image or video file")
    args = parser.parse_args()

    model = load_model()

    if args.path.lower().endswith((".jpg", ".jpeg", ".png")):
        predict_image(args.path, model)
    else:
        score = get_video_fake_score(args.path, model)
        level = interpret_fake_score(score)
        print(f"Video fake prob: {score:.3f} ({level})")
