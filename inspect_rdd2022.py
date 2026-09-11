import os
from pathlib import Path
from collections import Counter
from PIL import Image

# ============================================================
# RDD2022 DATASET INSPECTION SCRIPT
# ============================================================

# ------------------------------------------------------------
# 1. DATASET PATH
# ------------------------------------------------------------
# This script assumes:
#
# DEEP_LEARNING_PROJECT/
# ├── Dataset/
# │   └── RDD_SPLIT/
# │       ├── train/
# │       ├── val/
# │       └── test/
# └── inspect_rdd2022.py
#
# If your location is different, change this path.
# ------------------------------------------------------------

DATASET_PATH = Path("Dataset/RDD_SPLIT")

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Expected YOLO format:
# class_id x_center y_center width height


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_images(folder):
    """Return all image files inside a folder."""
    return [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_labels(folder):
    """Return all TXT label files inside a folder."""
    return [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() == ".txt"
    ]


def inspect_split(split_name):
    """Inspect one dataset split."""

    split_path = DATASET_PATH / split_name
    image_path = split_path / "images"
    label_path = split_path / "labels"

    print("\n" + "=" * 70)
    print(f"                    {split_name.upper()} SPLIT")
    print("=" * 70)

    if not image_path.exists():
        print(f"ERROR: Images folder not found: {image_path}")
        return None

    if not label_path.exists():
        print(f"ERROR: Labels folder not found: {label_path}")
        return None

    images = get_images(image_path)
    labels = get_labels(label_path)

    print(f"\nImages found : {len(images)}")
    print(f"Labels found : {len(labels)}")

    # --------------------------------------------------------
    # Image-label matching
    # --------------------------------------------------------

    image_stems = {img.stem for img in images}
    label_stems = {lbl.stem for lbl in labels}

    missing_labels = image_stems - label_stems
    extra_labels = label_stems - image_stems

    print("\n--- IMAGE / LABEL MATCHING ---")

    print(f"Images without labels : {len(missing_labels)}")
    print(f"Labels without images : {len(extra_labels)}")

    if missing_labels:
        print("\nFirst few images without labels:")
        for item in sorted(missing_labels)[:10]:
            print("  ", item)

    if extra_labels:
        print("\nFirst few labels without images:")
        for item in sorted(extra_labels)[:10]:
            print("  ", item)

    # --------------------------------------------------------
    # Check corrupted images
    # --------------------------------------------------------

    corrupted_images = []

    print("\n--- CHECKING IMAGE FILES ---")

    for img_file in images:
        try:
            with Image.open(img_file) as img:
                img.verify()
        except Exception:
            corrupted_images.append(img_file.name)

    print(f"Corrupted images : {len(corrupted_images)}")

    if corrupted_images:
        print("\nCorrupted images:")
        for item in corrupted_images[:10]:
            print("  ", item)

    # --------------------------------------------------------
    # Analyze labels
    # --------------------------------------------------------

    class_counter = Counter()

    total_annotations = 0
    invalid_labels = []
    empty_labels = []
    invalid_coordinates = []

    print("\n--- CHECKING YOLO LABELS ---")

    for label_file in labels:

        try:
            with open(label_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception:
            invalid_labels.append(label_file.name)
            continue

        # Empty label file
        if len(lines) == 0:
            empty_labels.append(label_file.name)
            continue

        for line_number, line in enumerate(lines, start=1):

            parts = line.split()

            # YOLO annotation must have 5 values
            if len(parts) != 5:
                invalid_labels.append(
                    f"{label_file.name} (line {line_number})"
                )
                continue

            try:
                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

            except ValueError:
                invalid_labels.append(
                    f"{label_file.name} (line {line_number})"
                )
                continue

            # Count class
            class_counter[class_id] += 1
            total_annotations += 1

            # Check YOLO coordinate ranges
            coordinates = [
                x_center,
                y_center,
                width,
                height
            ]

            if not all(0 <= value <= 1 for value in coordinates):
                invalid_coordinates.append(
                    f"{label_file.name} (line {line_number})"
                )

            # Width and height must be > 0
            if width <= 0 or height <= 0:
                invalid_coordinates.append(
                    f"{label_file.name} (line {line_number})"
                )

    # --------------------------------------------------------
    # Print label results
    # --------------------------------------------------------

    print(f"Total annotations : {total_annotations}")

    print(f"Empty label files : {len(empty_labels)}")

    print(f"Invalid label lines : {len(invalid_labels)}")

    print(f"Invalid bounding boxes : {len(invalid_coordinates)}")

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\n--- CLASS DISTRIBUTION ---")

    if class_counter:

        for class_id, count in sorted(class_counter.items()):
            print(f"Class {class_id} : {count} annotations")

    else:
        print("No valid annotations found.")

    return {
        "images": len(images),
        "labels": len(labels),
        "missing_labels": len(missing_labels),
        "extra_labels": len(extra_labels),
        "corrupted_images": len(corrupted_images),
        "annotations": total_annotations,
        "empty_labels": len(empty_labels),
        "invalid_labels": len(invalid_labels),
        "invalid_boxes": len(invalid_coordinates),
        "classes": class_counter,
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

print("\n")
print("=" * 70)
print("              RDD2022 DATASET INSPECTION")
print("=" * 70)

if not DATASET_PATH.exists():

    print("\nERROR!")
    print(f"Dataset folder not found:")
    print(DATASET_PATH)

    print("\nMake sure your folder structure is:")
    print("""
DEEP_LEARNING_PROJECT/
│
├── Dataset/
│   └── RDD_SPLIT/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       │
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       │
│       └── test/
│           ├── images/
│           └── labels/
│
└── inspect_rdd2022.py
""")

    exit()


# Inspect all three splits

results = {}

for split in ["train", "val", "test"]:
    results[split] = inspect_split(split)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n\n")
print("=" * 70)
print("                    FINAL SUMMARY")
print("=" * 70)

print("\n")

print(f"{'SPLIT':<10} {'IMAGES':<10} {'LABELS':<10} {'ANNOTATIONS':<15}")

print("-" * 50)

for split, result in results.items():

    if result:

        print(
            f"{split:<10} "
            f"{result['images']:<10} "
            f"{result['labels']:<10} "
            f"{result['annotations']:<15}"
        )


# ============================================================
# CLASS SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("                    ALL CLASSES")
print("=" * 70)

all_classes = Counter()

for result in results.values():

    if result:

        all_classes.update(result["classes"])

for class_id, count in sorted(all_classes.items()):

    print(f"Class {class_id} : {count} total annotations")


# ============================================================
# PROBLEM SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("                    PROBLEM CHECK")
print("=" * 70)

total_problems = 0

for split, result in results.items():

    if not result:
        continue

    print(f"\n{split.upper()}:")

    problems = {
        "Missing labels": result["missing_labels"],
        "Extra labels": result["extra_labels"],
        "Corrupted images": result["corrupted_images"],
        "Empty labels": result["empty_labels"],
        "Invalid labels": result["invalid_labels"],
        "Invalid bounding boxes": result["invalid_boxes"],
    }

    for problem, count in problems.items():

        print(f"  {problem:<25}: {count}")

        total_problems += count


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n")
print("=" * 70)
print("                       VERDICT")
print("=" * 70)

if total_problems == 0:

    print("""
NO DATASET PROBLEMS DETECTED.

Your dataset passed the basic structural,
image, and YOLO annotation checks.

NEXT STEPS:

1. Visualize bounding boxes.
2. Verify class names.
3. Create data.yaml.
4. Start model training.
""")

else:

    print(f"""
{total_problems} potential dataset problems were detected.

DO NOT START MODEL TRAINING YET.

Review the results above first.
We need to determine whether the detected
issues require correction.
""")


print("\nInspection completed.")