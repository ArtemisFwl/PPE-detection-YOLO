"""
Inference using YOLOv8 pretrained model on a single image.
Purpose: Verify local environment, YOLO installation, and inference pipeline
before using the custom trained helmet detection model.
"""

from ultralytics import YOLO
model=YOLO("yolov8n.pt")

model.predict(
    source="test.jpg",
    conf=0.4,
    save=True,
    project="outputs",
    name="image",
    exist_ok=True
)

