"""
cat_food_detection.py
----------------------
Computer vision project using YOLO (You Only Look Once) for detecting 
a cat's food plate and classifying whether it is empty or not.

Overview
--------
This script applies object detection techniques to monitor pet feeding 
activity. It identifies the plate region and analyzes food presence to 
determine if the plate needs refilling.

Characteristics
---------------
- Uses YOLO (CNN-based deep learning model) for real-time detection.
- Designed to be modular and easily extended.
- Independent of specific hardware or adapters.
- Aims for reusability across different applications.
"""

import torch
from ultralytics import YOLO


# -----------------------------------------------------
# 1. Environment validation
# -----------------------------------------------------

#Validate if your laptop can train with GPU acceleration
print(torch.__version__) #See torch version

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training will run on: {DEVICE}") #GPU is faste than CPU.

# -----------------------------------------------------
# 2. Load pre-trained YOLO model
# -----------------------------------------------------

print("Loading YOLO model...")
model = YOLO("yolov11s.pt")  # you can also use yolov8s.pt, yolov8m.pt, etc.
model.info()
print("Model loaded successfully!\n")

# -----------------------------------------------------
# 3. Training section
# -----------------------------------------------------
DATA_YAML = r"C:\Users\rmalon\Downloads\PYTHON SCRIPT\proyectos_personales\proyectos_locales\Computer vision\data.yaml"


# training the model
results = model.train(data=r"C:\Users\rmalon\Downloads\PYTHON SCRIPT\proyectos_personales\proyectos_locales\Computer vision\data.yaml", 
                      epochs=60, imgsz=640,device=DEVICE)

# -----------------------------------------------------
# 4. Testing the trained model
# -----------------------------------------------------
best_trained_model = YOLO(r"C:\Users\rmalon\Downloads\PYTHON SCRIPT\proyectos_personales\proyectos_locales\Computer vision\runs\detect\train2\weights\best.pt")

# -----------------------------------------------------
# 5. Real-time detection using webcam
# -----------------------------------------------------

#Use the webcam
#Detects objects in the validation images using your trained model,
# showing only results above 50% confidence.
try:
    real_time_results = model.predict(source=0, conf=0.5, show=True)
except KeyboardInterrupt:
    print("\nWebcam detection interrupted by user.")

