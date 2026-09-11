from ultralytics import YOLO
import os
import random
import shutil


def main():

    # ============================================================
    # PATHS
    # ============================================================

    MODEL_PATH = r"C:\ALL_PROJECTS\DEEP_LEARNING_PROJECT\runs\detect\runs\yolo26n_final\weights\best.pt"

    TEST_IMAGES = r"C:\ALL_PROJECTS\DEEP_LEARNING_PROJECT\Dataset\RDD_SPLIT\test\images"

    PROJECT = r"C:\ALL_PROJECTS\DEEP_LEARNING_PROJECT\runs\predictions"
    RUN_NAME = "yolo26n_test_samples"

    # ============================================================
    # SETTINGS
    # ============================================================

    NUMBER_OF_IMAGES = 10
    IMAGE_SIZE = 640
    CONFIDENCE = 0.20

    # ============================================================
    # CHECK FILES
    # ============================================================

    print("=" * 70)
    print("YOLO26n - ROAD DAMAGE PREDICTION")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not os.path.exists(TEST_IMAGES):
        raise FileNotFoundError(
            f"Test images folder not found:\n{TEST_IMAGES}"
        )

    # ============================================================
    # FIND TEST IMAGES
    # ============================================================

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    images = [
        os.path.join(TEST_IMAGES, f)
        for f in os.listdir(TEST_IMAGES)
        if f.lower().endswith(image_extensions)
    ]

    if len(images) == 0:
        raise RuntimeError("No test images found.")

    # Random but reproducible selection
    random.seed(42)
    selected_images = random.sample(
        images,
        min(NUMBER_OF_IMAGES, len(images))
    )

    print(f"\nTotal test images available: {len(images)}")
    print(f"Images selected for prediction: {len(selected_images)}")

    # ============================================================
    # LOAD MODEL
    # ============================================================

    print("\nLoading YOLO26n best model...")

    model = YOLO(MODEL_PATH)

    print("Model loaded successfully.")

    # ============================================================
    # PREDICTION
    # ============================================================

    print("\n" + "=" * 70)
    print("STARTING PREDICTION")
    print("=" * 70)

    results = model.predict(
        source=selected_images,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=0,
        save=True,
        save_txt=True,
        save_conf=True,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        verbose=True
    )

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)

    for i, result in enumerate(results, start=1):

        image_name = os.path.basename(result.path)

        number_of_detections = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        print(
            f"{i:02d}. {image_name} "
            f"-> {number_of_detections} detections"
        )

        if number_of_detections > 0:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                print(
                    f"       {class_name}: "
                    f"{confidence:.2f}"
                )

    # ============================================================
    # OUTPUT LOCATION
    # ============================================================

    output_directory = os.path.abspath(
        os.path.join(PROJECT, RUN_NAME)
    )

    print("\n" + "=" * 70)
    print("PREDICTION COMPLETED")
    print("=" * 70)

    print("\nAnnotated images saved to:")
    print(output_directory)

    print("\nThe folder contains:")
    print("- Annotated prediction images")
    print("- Prediction label files")
    print("- Confidence scores")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()