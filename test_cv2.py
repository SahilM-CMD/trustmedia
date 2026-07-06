import cv2

path = r"S:\Python\TrustMediaBackend\DeepfakeDetector\Test_Material\Video_Creation_Request_Fulfilled.mp4"
cap = cv2.VideoCapture(path)
print("Opened:", cap.isOpened())
print("Backend:", cap.getBackendName() if hasattr(cap, "getBackendName") else "unknown")
ret, frame = cap.read()
print("First frame read:", ret)
cap.release()
