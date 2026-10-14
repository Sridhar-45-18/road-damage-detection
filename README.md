# Road Damage Detection & Classification (RoadScan AI)

Road damage detection and classification using **YOLO26n** trained on the **RDD2022** dataset, featuring an interactive web application for image detection, video analysis, and real-time live camera feed.

## Features
- **Image Detection**: Side-by-side comparison of original vs detected road damages with overlapping bounding boxes support.
- **Video Analysis**: Frame-by-frame detection with automated browser-compatible H.264 video encoding.
- **Live Camera Feed**: Real-time 60 FPS camera capture with continuous road defect detection and performance metrics.
- **Damage Classes (RDD2022)**:
  - `Longitudinal Crack` (Class 0)
  - `Transverse Crack` (Class 1)
  - `Alligator Crack` (Class 2)
  - `Other Corruption` (Class 3)
  - `Pothole` (Class 4)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
python webapp/app.py
```
Open **`http://localhost:5000`** in your browser.
