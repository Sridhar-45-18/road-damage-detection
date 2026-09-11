from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

IMAGE_DIR = Path("Dataset/RDD_SPLIT/train/images")
LABEL_DIR = Path("Dataset/RDD_SPLIT/train/labels")

NUM_IMAGES = 12

# Your RDD2022 Kaggle dataset has 5 classes
CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Other Corruption",
    4: "Pothole"
}


# ============================================================
# FIND IMAGES
# ============================================================

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in image_extensions
]

if not images:
    print("No images found!")
    exit()

# Select random images
random.seed(42)

sample_images = random.sample(
    images,
    min(NUM_IMAGES, len(images))
)


# ============================================================
# VISUALIZE
# ============================================================

for image_path in sample_images:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    # Read image
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path.name}")
        continue

    # Convert BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_height, image_width = image.shape[:2]

    # --------------------------------------------------------
    # Read YOLO labels
    # --------------------------------------------------------

    if label_path.exists():

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert normalized YOLO coordinates
            # to pixel coordinates

            x_center *= image_width
            y_center *= image_height

            width *= image_width
            height *= image_height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)

            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_width - 1, x2)
            y2 = min(image_height - 1, y2)

            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Class name
            class_name = CLASS_NAMES.get(
                class_id,
                f"Unknown ({class_id})"
            )

            label_text = f"{class_id}: {class_name}"

            # Draw label
            cv2.putText(
                image,
                label_text,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    plt.figure(figsize=(12, 7))

    plt.imshow(image)

    plt.title(image_path.name)

    plt.axis("off")

    plt.show()


print("\nVisualization completed.")