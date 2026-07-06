import os
import sys
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(__file__))
from Classifier import get_model  # note capital C if your file is Classifier.py

IMAGE_PATH = r"S:\Python\TrustMediaBackend\DeepfakeDetector\Testing_Material\RealFace.jpg"

# ---- copy Android preprocessing exactly ----
# 1) load image as RGB (this is equivalent to your Bitmap)
img = Image.open(IMAGE_PATH).convert("RGB")

# 2) resize to 224x224 with bilinear interpolation
img_resized = img.resize((224, 224), resample=Image.BILINEAR)

# 3) convert to tensor and normalize with same mean/std
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img_resized)

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_tensor = (img_tensor - mean) / std

x = img_tensor.unsqueeze(0)  # [1,3,224,224]

model = get_model()
model.eval()

with torch.no_grad():
    logit = model(x)[0, 0].item()
    fake = torch.sigmoid(torch.tensor(logit)).item()

print("android_style_logit =", logit)
print("android_style_fake  =", fake)
