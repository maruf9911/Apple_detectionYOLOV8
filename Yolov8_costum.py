from ultralytics import YOLO

# Load a pretrained Yolov8n model
model = YOLO("venv/best.pt")


# Run inference on the source
results = model(source=0, show=True, conf=0.3, save=True) # generator of Result objects
