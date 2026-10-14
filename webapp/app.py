"""
Road Damage Detection Web Application - Flask Backend
Uses YOLO26n model trained on RDD2022 dataset.
Detects: Longitudinal Crack, Transverse Crack, Alligator Crack,
         Other Corruption, Pothole
"""

from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, Response
)
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid
import time
import threading
import base64
import json

# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)
CORS(app)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results_web")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Model path - use the best trained weights
# webapp/ is inside the project root, so go up one level
PROJECT_ROOT = os.path.dirname(BASE_DIR)
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "best.pt")

# Fallback: try yolo26n.pt in root
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, "yolo26n.pt")

# Class names and colors (matching data.yaml)
CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Other Corruption",
    4: "Pothole"
}

# Vibrant, distinct colors for each class (BGR format for OpenCV)
CLASS_COLORS = {
    0: (66, 165, 245),    # Blue
    1: (239, 108, 0),     # Orange
    2: (102, 187, 106),   # Green
    3: (255, 213, 79),    # Amber
    4: (229, 57, 53),     # Red
}

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("Loading YOLO26n Road Damage Detection Model...")
print(f"Model path: {MODEL_PATH}")
print("=" * 60)

model = None
try:
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("The application will start but detection won't work.")

# ============================================================
# DETECTION SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.10
IMAGE_SIZE = 640

# Global camera state
camera_active = False
camera_lock = threading.Lock()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def draw_detections(image, results):
    """Draw bounding boxes and labels on image. Allow overlapping boxes."""
    annotated = image.copy()
    detections = []

    if results and len(results) > 0:
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")

                # Color for this class (BGR)
                color = CLASS_COLORS.get(class_id, (200, 200, 200))

                # Draw filled rectangle background for label
                label = f"{class_name} {confidence:.0%}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness = 2
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                # Draw label background
                label_y = max(y1 - 10, label_h + 10)
                cv2.rectangle(
                    annotated,
                    (x1, label_y - label_h - baseline - 5),
                    (x1 + label_w + 5, label_y + 5),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    annotated, label,
                    (x1 + 2, label_y - 3),
                    font, font_scale, (255, 255, 255), thickness
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [x1, y1, x2, y2]
                })

    return annotated, detections


def run_detection(image):
    """Run YOLO detection on image."""
    if model is None:
        return image, []

    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device="cpu",
        verbose=False,
        agnostic_nms=False,  # Allow overlapping boxes across different classes
        iou=0.90,            # High IoU threshold so overlapping boxes are not suppressed/restricted
        max_det=300
    )

    annotated, detections = draw_detections(image, results)
    return annotated, detections


# ============================================================
# ROUTES - PAGES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/image")
def image_page():
    return render_template("image.html")


@app.route("/video")
def video_page():
    return render_template("video.html")


@app.route("/live")
def live_page():
    return render_template("live.html")


# ============================================================
# ROUTES - IMAGE DETECTION API
# ============================================================

@app.route("/api/detect/image", methods=["POST"])
def detect_image():
    """Handle image upload and return detection results."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image file"}), 400

        # Save original
        uid = str(uuid.uuid4())[:8]
        original_name = f"original_{uid}.jpg"
        detected_name = f"detected_{uid}.jpg"

        original_path = os.path.join(RESULT_DIR, original_name)
        detected_path = os.path.join(RESULT_DIR, detected_name)

        cv2.imwrite(original_path, image)

        # Run detection
        start_time = time.time()
        annotated, detections = run_detection(image)
        inference_time = round((time.time() - start_time) * 1000, 1)

        cv2.imwrite(detected_path, annotated)

        # Summary
        damage_summary = {}
        for d in detections:
            name = d["class_name"]
            if name not in damage_summary:
                damage_summary[name] = 0
            damage_summary[name] += 1

        return jsonify({
            "success": True,
            "original_url": f"/results/{original_name}",
            "detected_url": f"/results/{detected_name}",
            "detections": detections,
            "total_detections": len(detections),
            "damage_summary": damage_summary,
            "inference_time_ms": inference_time
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ROUTES - VIDEO DETECTION API
# ============================================================

@app.route("/api/detect/video", methods=["POST"])
def detect_video():
    """Handle video upload, process frame by frame, return detected video."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        uid = str(uuid.uuid4())[:8]
        input_name = f"input_{uid}.mp4"
        output_name = f"output_{uid}.mp4"

        input_path = os.path.join(UPLOAD_DIR, input_name)
        output_path = os.path.join(RESULT_DIR, output_name)

        file.save(input_path)

        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video file"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output writer - use mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_detections = 0
        all_damage_types = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, detections = run_detection(frame)
            out.write(annotated)

            frame_count += 1
            total_detections += len(detections)

            for d in detections:
                name = d["class_name"]
                if name not in all_damage_types:
                    all_damage_types[name] = 0
                all_damage_types[name] += 1

        cap.release()
        out.release()

        # Convert to browser-compatible H.264 format using ffmpeg
        web_output_name = f"web_{uid}.mp4"
        web_output_path = os.path.join(RESULT_DIR, web_output_name)

        ffmpeg_success = False
        try:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg_exe = "ffmpeg"

            import subprocess
            result = subprocess.run(
                [
                    ffmpeg_exe, "-y", "-i", output_path,
                    "-vcodec", "libx264", "-acodec", "aac",
                    "-movflags", "+faststart",
                    "-pix_fmt", "yuv420p",
                    web_output_path
                ],
                capture_output=True, timeout=300
            )
            if result.returncode == 0 and os.path.exists(web_output_path):
                ffmpeg_success = True
                try:
                    os.remove(output_path)
                except Exception:
                    pass
        except Exception as e:
            print(f"FFmpeg conversion error: {e}")

        video_url = f"/results/{web_output_name}" if ffmpeg_success else f"/results/{output_name}"

        # Clean up input
        try:
            os.remove(input_path)
        except Exception:
            pass

        return jsonify({
            "success": True,
            "video_url": video_url,
            "total_frames": frame_count,
            "total_detections": total_detections,
            "damage_summary": all_damage_types,
            "fps": fps,
            "resolution": f"{width}x{height}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ROUTES - LIVE CAMERA FEED
# ============================================================

@app.route("/api/detect/frame", methods=["POST"])
def detect_frame():
    """Detect damage in a single frame sent from the browser camera."""
    try:
        data = request.get_json()
        if not data or "frame" not in data:
            return jsonify({"error": "No frame data"}), 400

        # Decode base64 image
        frame_data = data["frame"]
        # Remove data URL prefix if present
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]

        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid frame"}), 400

        # Run detection
        start = time.time()
        annotated, detections = run_detection(frame)
        inference_ms = round((time.time() - start) * 1000, 1)

        # Encode result back to base64
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        result_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "success": True,
            "frame": f"data:image/jpeg;base64,{result_b64}",
            "detections": detections,
            "inference_ms": inference_ms
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ROUTES - SERVE RESULTS
# ============================================================

@app.route("/results/<filename>")
def serve_result(filename):
    return send_from_directory(RESULT_DIR, filename)


# ============================================================
# MODEL INFO API
# ============================================================

@app.route("/api/model-info")
def model_info():
    """Return model information."""
    if model is None:
        return jsonify({"loaded": False})

    return jsonify({
        "loaded": True,
        "model_path": MODEL_PATH,
        "classes": CLASS_NAMES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "image_size": IMAGE_SIZE
    })


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Road Damage Detection Web Application")
    port = int(os.environ.get("PORT", 5000))
    print(f"  Open: http://localhost:{port}")
    print("=" * 60 + "\n")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )
