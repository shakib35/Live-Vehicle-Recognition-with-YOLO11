import os
from ultralytics import YOLO
import comet_ml
from comet_ml import start
import torch
import time

# Comet environment variables
os.environ["COMET_EVAL_BATCH_LOGGING_INTERVAL"] = "5"
os.environ["COMET_EVAL_LOG_CONFUSION_MATRIX"] = "false"

COMET_SUPPORTED_TASKS = ["detect"]
EVALUATION_PLOT_NAMES = "F1_curve", "P_curve", "R_curve", "PR_curve", "confusion_matrix"
LABEL_PLOT_NAMES = "labels", "labels_correlogram"

# Comet logging setup
experiment = start(
    api_key="W2y4B8fSkwtOQPElr8nni3A1H",
    project_name="YoLov11_Car_Object_Detection",
    workspace="shakib35"
)
experiment_name = "GPU test"
experiment.set_name(experiment_name)

# Parameters
device = "gpu"
model = YOLO("yolo11n.pt")

# Logging Parameters to Comet
experiment.log_parameter("device", device)
experiment.log_parameter("epochs", 10)

start_time = time.time()

# Train model based on YAML configuration
results = model.train(
    data="config.yaml", 
    project="YoLov11_Car_Object_Detection",
    save_period=1,
    batch=64,
    epochs=10,
    device=device
)

# Print runtime
end_time = time.time()
training_time = end_time - start_time
print(f"Training completed in {training_time} seconds")
experiment.log_metric(name="training_time", value=training_time)

# Save final trained model
model_path = 'Car_Object_Detection_cpu'
model.save(model_path)
print(f"Model saved to {model_path}")
experiment.log_model("trained_model", model_path)

# Evaluate on validation data to calculate metrics
metrics = model.val()

# Log validation metrics


# End the experiment
experiment.end()
