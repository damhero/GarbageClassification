from ultralytics import YOLO

model = YOLO("yolo11n run/runs/detect/train/weights/best.onnx")
results = model("path/to/image.jpg")  # Predict on an image
results[0].show()  # Display results