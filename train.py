from ultralytics import YOLO
from comet_ml import start
import torch
import time


# Comet logging setup
experiment = start(
    api_key = "W2y4B8fSkwtOQPElr8nni3A1H",
    project_name = "YoLov11_Car_Object_Detection",
    workspace="shakib35"
)
# Parameters
device = "mps"
model = YOLO("yolo11n.pt")

# Logging Parameters to Comet
experiment.log_parameter("device", device)
experiment.log_parameter("epochs", 10)


start_time = time.time()
# train model based on yaml
results = model.train(data="config.yaml", epochs=10, device=device)

# print run time
end_time = time.time()
training_time = end_time - start_time
print(f"Training completed in {training_time} seconds")
experiment.log_metric("training_time ", training_time)

# Save final trained model
model_path = 'Car_Object_Detection_mps'
model.save(model_path)
print(f"Model saved to {model_path}")
experiment.log_model("trained_model", model_path)

# Evaluate on validation data to calculate metrics
metrics = model.val()

# Log evaluation metrics to Comet
experiment.log_metric("precision", metrics.box.precision.mean())
experiment.log_metric("recall", metrics.box.recall.mean())
experiment.log_metric("mAP_50", metrics.box.map50)
experiment.log_metric("mAP_50_95", metrics.box.map)

# Log a Precision-Confidence Curve to Comet
precision_confidence_curve = metrics.box.pr_curve  # Assuming this is supported by YOLO
experiment.log_curve("Precision-Confidence Curve", precision_confidence_curve.confidence, precision_confidence_curve.precision)

# Log Confusion Matrix to Comet
confusion_matrix = metrics.box.confusion_matrix
experiment.log_confusion_matrix(matrix=confusion_matrix, labels=metrics.names)

# End the experiment
experiment.end()