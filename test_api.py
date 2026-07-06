import requests
import json

url = "http://localhost:5000/detect-image"
files = {'file': open(r"S:\Python\TrustMediaBackend\DeepfakeDetector\Testing_Material\RealFace.jpg", 'rb')}

response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("\nJSON Response:")
print(json.dumps(response.json(), indent=2))
