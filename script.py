!pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless && pip install -qU "opencv-python-headless==4.10.*" "numpy<2.0"
!pip install -qU ultralytics --no-deps
from ultralytics import YOLO
# ^ fixing kaggle dependencies

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# Train the model on the dataset dataset change data="path" to your path
train_results = model.train(
    data="TU/WPISZ/SWOJA/ŚCIEŻKĘ",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    patience=10 #early stopping after 10 epochs without improvement
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model
