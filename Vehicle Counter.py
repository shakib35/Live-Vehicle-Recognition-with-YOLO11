from ultralytics import YOLO
import cv2
import numpy as np
import torch

# Load YOLO model
device = 'cpu'
model_path = r"B:\Documents\GitHub\Live-Vehicle-Recognition-with-YOLO11\YoLov11_Car_Object_Detection\train8\weights\best.pt"
model = YOLO(model_path).to(device)


VEHICLE_CLASSES = ["vehicle"]

# Initialize vehicle counting variables
vehicle_count = 0
line_position = 300  # Horizontal line position for counting
counted_ids = set()  # Set to track counted vehicle IDs

def process_frame(frame, original_frame):
    global vehicle_count, counted_ids

    gray_frame_3channel = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Perform inference on the 3-channel grayscale frame
    results = model(gray_frame_3channel)

    # Extract detections
    detections = results[0].boxes
    for detection in detections:
        box = detection.xyxy[0].cpu().numpy()  # Bounding box [x1, y1, x2, y2]
        score = detection.conf.cpu().item()  # Confidence score
        class_id = int(detection.cls.cpu().item())  # Class ID

        # Check if the detected object is a vehicle
        if score > 0.5 and VEHICLE_CLASSES[class_id] in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box)

            # Draw bounding box and label on the original color frame
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{VEHICLE_CLASSES[class_id]} {score:.2f}"
            cv2.putText(original_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Check if the vehicle crosses the counting line
            center_y = (y1 + y2) // 2
            if line_position - 5 < center_y < line_position + 5:
                vehicle_id = f"{x1}-{y1}-{x2}-{y2}"  # Use bounding box as a unique ID
                if vehicle_id not in counted_ids:
                    vehicle_count += 1
                    counted_ids.add(vehicle_id)

    # Draw counting line and count on the original color frame
    height, width, _ = original_frame.shape
    cv2.line(original_frame, (0, line_position), (width, line_position), (0, 255, 0), 2)
    cv2.putText(original_frame, f"Count: {vehicle_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return original_frame

def process_video(video_path, output_path):
    global vehicle_count, counted_ids
    vehicle_count = 0
    counted_ids = set()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize VideoWriter to save the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, original_frame = cap.read()
        if not ret:
            break

        # Convert the original frame to grayscale for processing
        gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

        # Process the grayscale frame and return the annotated color frame
        processed_frame = process_frame(gray_frame, original_frame)

        # Write the processed frame to the output video
        out.write(processed_frame)

        # Display the processed color frame (optional)
        cv2.imshow("Vehicle Detection and Counting", processed_frame)

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Main function
video_path = r"path/to/your/video.mp4"  # Replace with your video file path
output_path = r"path/to/output/video.mp4"  # Replace with your desired output file path
process_video(video_path, output_path)
