# Food plate Detector — Project Documentation

> **Description:** Real-time computer vision system that detects a cat's plate and food inside the plate. Uses YOLO (CNN-based object detection) for perception and simple area-based logic for decision.

---

## Table of Contents

1. Project overview
2. Goals & success criteria
3. Quickstart (run locally)
4. Folder structure
5. Dataset & labeling
6. Models and training
7. Evaluation & metrics
8. Deployment & inference
9. Dependencies
10. Development workflow & git
11. Documentation standards (public + internal)
12. Internal notes template (how to use)
13. Troubleshooting & common issues
14. Roadmap & TODOs
15. References & resources

---

## 1. Project overview

**What it does**
- Detects `plate` and `food` in images/video frames.

**Why it exists**
- Practical home automation (notify when cat needs food).
- Portfolio + learning project merging labeling, model training and deployment.

---

## 2. Goals & success criteria

**Goals**
- Working prototype on laptop with webcam.
- Train YOLO on custom dataset (`plate`, `food`).

**Success criteria**
- Plate detected in >80% of frames on validation set.

---

## 3. Quickstart (run locally)

> Assumes a Python 3.11+ environment. Prefer conda.

```bash
# create environment
conda create --name cvision Python = 3.12 #Windows cmd

# install dependencies
pip install -r requirements.txt

# run inference on webcam
python cat_food_detector.py
```

---

## 4. Folder structure (recommended)

```
cat-food-detector/
├── cat_food_detector_documentation.md
├── data/
│   ├── train/
│   ├── validation/
├── models/
│   ├── best.pt/				# best trained model
│   ├── yolo11s.pt/			# pretrained model yolo 11 small
├── src/
│   ├── cat_food_detector.py  # wrapper for running YOLO + postprocessing
├── requirements.txt
├── data.yml
```

---

## 5. Dataset & labeling

**Objects / classes:** `plate`, `food`.

**Image types to include:**
- Plate full of food (many variations of amount and food types)
- Plate half full / partially eaten
- Empty plate (no food)
- Occasionally frames with no plate (background) to increase robustness

**Labeling tool:** Label Studio.

**Label format (YOLO):** For each image `image.jpg` there is `image.txt` with lines:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to [0,1] relative to image width/height.

**Dataset tips:**
- Capture different distances, angles, lighting, backgrounds.
- Augment images (rotation, brightness, flip) to increase diversity.
- Keep separate `train/val/test` splits and avoid duplicates across splits.

---

## 6. Models and training

**Model choice:** YOLO11s(Ultralytics) recommended for ease of use.

**Training steps (high level):**
1. Prepare `data.yaml` pointing to `train` and `val` images and class names.
2. Configure hyperparameters (batch size, lr, epochs, image size).
3. Start training with "results = model.train(data=DATA_YAML_PATH,epochs=60, imgsz=640,device=DEVICE)".


---

## 9. Dependencies

**Core dependencies** (example `requirements.txt`):
```
python>=3.11
torch>=1.13  # match your CUDA or CPU
opencv-python
ultralytics  #
numpy
pandas
label-studio
```
---

## 10. Development workflow & git

**Recommended branching:**
- `main` (stable releases and working demos)
- `dev` (active development)

**Commit style:** short summary

**Dataset & model versioning:**
- Keep datasets and labels under a separate storage (do not commit large binaries to git)
- Use `git-lfs` for model weights if you must store them: git lfs track "*.jpg", "*.png", etc.

---
