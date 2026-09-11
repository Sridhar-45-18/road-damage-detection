from ultralytics import YOLO
import torch
import os


def main():

    # ============================================================
    # PATHS
    # ============================================================

    MODEL_PATH = r"C:\ALL_PROJECTS\DEEP_LEARNING_PROJECT\runs\detect\runs\yolo26n_final\weights\best.pt"
    DATASET_PATH = r"C:\ALL_PROJECTS\DEEP_LEARNING_PROJECT\data.yaml"

    # Separate folder for final test evaluation
    PROJECT = r"C:\ALL_PROJECTS\DEEP_LEARNING_PROJECT\runs\test_evaluation"
    RUN_NAME = "yolo26n_test"

    # ============================================================
    # BASIC CHECKS
    # ============================================================

    print("=" * 70)
    print("YOLO26n - FINAL TEST EVALUATION")
    print("=" * 70)

    print("\nHardware Information")
    print("-" * 70)

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is NOT available. Evaluation has been stopped."
        )

    print("GPU:", torch.cuda.get_device_name(0))

    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(f"VRAM: {gpu_memory:.1f} GB")
    print("CUDA version:", torch.version.cuda)

    # ============================================================
    # CHECK FILES
    # ============================================================

    print("\nChecking required files")
    print("-" * 70)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\nBest model not found:\n{MODEL_PATH}"
        )

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"\nDataset YAML not found:\n{DATASET_PATH}"
        )

    print("Model:", MODEL_PATH)
    print("Dataset:", DATASET_PATH)

    # ============================================================
    # LOAD MODEL
    # ============================================================

    print("\nLoading trained YOLO26n model...")
    model = YOLO(MODEL_PATH)

    print("Model loaded successfully.")

    # ============================================================
    # FINAL TEST EVALUATION
    # ============================================================

    print("\n" + "=" * 70)
    print("STARTING FINAL TEST EVALUATION")
    print("=" * 70)

    metrics = model.val(
        data=DATASET_PATH,

        # IMPORTANT:
        # Evaluate the TEST split, not validation
        split="test",

        imgsz=640,
        batch=8,
        device=0,
        workers=0,

        # Save evaluation results
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,

        # Generate plots
        plots=True,

        verbose=True
    )

    # ============================================================
    # PRINT FINAL RESULTS
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"\nPrecision      : {metrics.box.mp:.4f}")
    print(f"Recall         : {metrics.box.mr:.4f}")
    print(f"mAP@50         : {metrics.box.map50:.4f}")
    print(f"mAP@50-95      : {metrics.box.map:.4f}")

    print("\nPercentage form:")

    print(f"Precision      : {metrics.box.mp * 100:.2f}%")
    print(f"Recall         : {metrics.box.mr * 100:.2f}%")
    print(f"mAP@50         : {metrics.box.map50 * 100:.2f}%")
    print(f"mAP@50-95      : {metrics.box.map * 100:.2f}%")

    # ============================================================
    # PER-CLASS RESULTS
    # ============================================================

    print("\n" + "=" * 70)
    print("PER-CLASS TEST RESULTS")
    print("=" * 70)

    class_names = model.names

    for i, name in class_names.items():

        print(f"\n{name}")

        print(
            f"  Precision : {metrics.box.p[i]:.4f}"
        )

        print(
            f"  Recall    : {metrics.box.r[i]:.4f}"
        )

        print(
            f"  mAP@50    : {metrics.box.ap50[i]:.4f}"
        )

        print(
            f"  mAP@50-95 : {metrics.box.ap[i]:.4f}"
        )

    # ============================================================
    # RESULT LOCATION
    # ============================================================

    print("\n" + "=" * 70)
    print("TEST EVALUATION COMPLETED")
    print("=" * 70)

    print("\nEvaluation results saved to:")

    print(
        os.path.abspath(
            os.path.join(PROJECT, RUN_NAME)
        )
    )

    print("\nImportant files to look for:")

    print("- confusion_matrix.png")
    print("- confusion_matrix_normalized.png")
    print("- BoxP_curve.png")
    print("- BoxR_curve.png")
    print("- BoxF1_curve.png")
    print("- BoxPR_curve.png")

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION FINISHED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()