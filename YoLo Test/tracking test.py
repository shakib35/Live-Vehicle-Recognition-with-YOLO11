from ultralytics import YOLO
import cv2

# Path to the YOLO model and video file
model_path = r"B:\Documents\GitHub\Live-Vehicle-Recognition-with-YOLO11\Car Detection\medium\weights\best.pt"
video_path = r"E:\vehicles.mp4"
# Load the YOLO model
model = YOLO(model_path)

results = model.track(video_path, show=True, tracker='bytetrack.yaml')