from ultralytics import YOLO
import cv2
import torch

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Check if MPS is available, otherwise fallback to CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"


results_train = model.train(data="coco8.yaml", epochs=10, imgsz=640, device=device)

results_inference = model.predict(source='0', show=True)
print(results_inference)

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=20, imgsz=640, device=device)


