from ultralytics import YOLO
import cv2

# Path to the YOLO model and input image
model_path = r'B:\Documents\GitHub\Live-Vehicle-Recognition-with-YOLO11\YoLov11_Car_Object_Detection\train2\weights\best.pt'
image_path = r"B:\Downloads\car.jpg"  # Replace with the path to your image
output_path = r"B:\Downloads\car_annotated.jpg"  # Replace with the desired output path

# Load the YOLO model
model = YOLO(model_path)

# Load the image
image = cv2.imread(image_path)

# Ensure the image is loaded
if image is None:
    print(f"Error: Unable to load image from {image_path}")
    exit()

# Run inference on the image
results = model.predict(source=image, save=False)

# Extract results
boxes = results[0].boxes  # Detected bounding boxes
for box in boxes:
    # Bounding box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    # Confidence score
    confidence = box.conf.cpu().item()
    # Class ID
    class_id = int(box.cls.cpu().item())
    # Class name (you may need to map class_id to actual names depending on your model's labels)
    label = f"{model.names[class_id]} {confidence:.2f}"

    # Draw bounding box and label on the image
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Save the output image
cv2.imwrite(output_path, image)
print(f"Output saved to {output_path}")

# Display the output image (optional)
cv2.imshow("Detected Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
