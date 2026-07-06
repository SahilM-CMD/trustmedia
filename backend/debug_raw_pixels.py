from PIL import Image
from torchvision import transforms
import torch

IMAGE_PATH = r"S:\Python\TrustMediaBackend\DeepfakeDetector\Testing_Material\Fake_Face.png"

img = Image.open(IMAGE_PATH).convert("RGB")
img_resized = img.resize((224, 224), Image.BILINEAR)

# NO normalization, just to tensor
to_tensor = transforms.ToTensor()
x = to_tensor(img_resized)  # shape [3, 224, 224], 0..1 range

# Flatten and print first 10 pixels
flat = x.view(-1)
print("Raw pixels (first 10):", flat[:10].tolist())
