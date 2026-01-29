from ultralytics import YOLO
model=YOLO("yolov8n.pt")

results = model("test.jpg", conf=0.4, save=True)

model= YOLO("weights/best.pt")

print(model.names)
