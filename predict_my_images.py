from ultralytics import YOLO
import os


def main():

    MODEL_PATH = r"C:\ALL_PROJECTS\DEEP_LEARNING_PROJECT\runs\detect\runs\yolo26n_final\weights\epoch49.pt"

    SOURCE = r"C:\ALL_PROJECTS\DEEP_LEARNING_PROJECT\my_images"

    OUTPUT_PROJECT = r"C:\ALL_PROJECTS\DEEP_LEARNING_PROJECT\runs\my_predictions"
    OUTPUT_NAME = "yolo26n_custom_images"

    print("=" * 70)
    print("YOLO26n - CUSTOM IMAGE PREDICTION")
    print("=" * 70)

    # Check files
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

    if not os.path.exists(SOURCE):
        raise FileNotFoundError(f"Image folder not found:\n{SOURCE}")

    # Load trained model
    print("\nLoading trained YOLO26n...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")

    # Run prediction
    print("\nStarting prediction...")

    results = model.predict(
        source=SOURCE,
        imgsz=640,
        conf=0.10,
        device=0,
        save=True,
        save_txt=True,
        save_conf=True,
        project=OUTPUT_PROJECT,
        name=OUTPUT_NAME,
        exist_ok=True,
        verbose=True
    )

    # Print detections
    print("\n" + "=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)

    for result in results:

        image_name = os.path.basename(result.path)

        print(f"\nImage: {image_name}")

        if result.boxes is None or len(result.boxes) == 0:
            print("  No damage detected.")
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            print(
                f"  {class_name} "
                f"(confidence: {confidence:.2f})"
            )

    output_directory = os.path.abspath(
        os.path.join(OUTPUT_PROJECT, OUTPUT_NAME)
    )

    print("\n" + "=" * 70)
    print("PREDICTION COMPLETED")
    print("=" * 70)

    print("\nResults saved to:")
    print(output_directory)


if __name__ == "__main__":
    main()