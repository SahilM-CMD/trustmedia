import os
import sys
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(__file__))

IMAGE_PATH = r"S:\Python\TrustMediaBackend\DeepfakeDetector\Testing_Material\RealFace.jpg"
TS_MODEL_PATH = r"S:\Python\TrustMediaBackend\DeepfakeDetector\efficientnet_ffpp_best_mobile.pt"

# same preprocessing as Step 1
img = Image.open(IMAGE_PATH).convert("RGB")
img_resized = img.resize((224, 224), resample=Image.BILINEAR)
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img_resized)

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_tensor = (img_tensor - mean) / std

x = img_tensor.unsqueeze(0)  # [1,3,224,224]

ts_model = torch.jit.load(TS_MODEL_PATH, map_location="cpu")
ts_model.eval()

with torch.no_grad():
    ts_output = ts_model(x)[0, 0].item()
    ts_fake = torch.sigmoid(torch.tensor(ts_output)).item()

print("ts_logit =", ts_output)
print("ts_fake  =", ts_fake)
