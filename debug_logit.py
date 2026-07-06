import os
import sys
from PIL import Image
import torch
from torchvision import transforms

# make sure we can import classifier.py from this folder
sys.path.append(os.path.dirname(__file__))

from Classifier import get_model

IMAGE_PATH = r"S:\Python\TrustMediaBackend\DeepfakeDetector\Testing_Material\RealFace.jpg"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = get_model()
model.eval()


img = Image.open(IMAGE_PATH).convert("RGB")
x = transform(img).unsqueeze(0)

with torch.no_grad():
    logit = model(x)[0, 0].item()
    fake = torch.sigmoid(torch.tensor(logit)).item()

print("python_logit =", logit)
print("python_fake  =", fake)

