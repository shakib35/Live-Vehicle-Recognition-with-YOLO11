from ultralytics import YOLO
import torch

device = "cuda"

model = YOLO("yolov11n.pt")

results = model.train(data="config.yaml", epochs=1, device=device)