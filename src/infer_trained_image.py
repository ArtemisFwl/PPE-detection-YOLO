"""
Inference using custom trained YOLOv8 helmet detection model.
Purpose: Run prediction on a single image using the final trained model (best.pt).
"""
from ultralytics import YOLO
# Load the trained model

model=YOLO("weights/trained/best.pt")

# Run inference on an Image 
model.predict(
    source="data/test.jpg",
    save=True,
    conf=0.4,
    project="outputs/trained/images",
    name="trained_image"

)