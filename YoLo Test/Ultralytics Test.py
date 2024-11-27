from ultralytics import YOLO
import cv2

# Path to the YOLO model and video file
model_path = '/Users/shaki/Documents/GitHub/Live-Vehicle-Recognition-with-YOLO11/YoLov11_Car_Object_Detection/train2/weights/best.pt'
video_path = '/Volumes/T7/vehicles.mp4'

# Load the YOLO model
model = YOLO(model_path)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    exit()

# Process the video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale back to 3-channel format for YOLO compatibility
    gray_frame_3channel = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # Run inference on the grayscale frame
    results = model.predict(source=gray_frame_3channel, save=False, show=True)

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
