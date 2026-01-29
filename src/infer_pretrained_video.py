"""
Video inference using YOLOv8 pretrained model.
Purpose: Verify video inference pipeline and performance before using trained model.
"""

from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Run inference on video
model.predict(
    source="data/test_video.mp4",
    save=True,
    conf=0.4,
    project="outputs",
    name="pretrained_video"
)
