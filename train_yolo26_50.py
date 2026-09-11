from ultralytics import YOLO
import torch
import os
import time


def main():

    # ==========================================================
    # CONFIGURATION
    # ==========================================================

    MODEL = "yolo26n.pt"
    DATASET = os.path.abspath("./data.yaml")

    EPOCHS = 50
    IMAGE_SIZE = 640
    BATCH_SIZE = 8

    DEVICE = 0
    WORKERS = 0

    PROJECT = "./runs"
    RUN_NAME = "yolo26n_final"

    # ==========================================================
    # HEADER
    # ==========================================================

    print("=" * 70)
    print("YOLO26n - RDD2022 FINAL TRAINING")
    print("=" * 70)

    # ==========================================================
    # GPU CHECK
    # ==========================================================

    print("\nHardware Information")
    print("-" * 70)

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "\nCUDA is NOT available.\n"
            "Training has been stopped to prevent CPU training."
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print("GPU:", gpu_name)
    print(f"VRAM: {gpu_memory:.1f} GB")
    print("CUDA version:", torch.version.cuda)

    # ==========================================================
    # DATASET CHECK
    # ==========================================================

    print("\nDataset Information")
    print("-" * 70)

    if not os.path.exists(DATASET):
        raise FileNotFoundError(
            f"\nDataset configuration not found:\n{DATASET}"
        )

    print("Dataset:", DATASET)

    # ==========================================================
    # TRAINING CONFIGURATION
    # ==========================================================

    print("\nTraining Configuration")
    print("-" * 70)

    print("Model:", MODEL)
    print("Epochs:", EPOCHS)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Device: GPU 0")
    print("Workers:", WORKERS)
    print("AMP: True")
    print("Pretrained: True")

    # ==========================================================
    # LOAD MODEL
    # ==========================================================

    print("\nLoading YOLO26n...")

    model = YOLO(MODEL)

    print("YOLO26n loaded successfully.")

    # ==========================================================
    # START TRAINING
    # ==========================================================

    print("\n" + "=" * 70)
    print("STARTING FINAL TRAINING")
    print("=" * 70)

    start_time = time.time()

    results = model.train(

        # ------------------------------------------------------
        # Dataset
        # ------------------------------------------------------
        data=DATASET,

        # ------------------------------------------------------
        # Main training
        # ------------------------------------------------------
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        # ------------------------------------------------------
        # GPU
        # ------------------------------------------------------
        device=DEVICE,

        # ------------------------------------------------------
        # Windows-safe data loading
        # ------------------------------------------------------
        workers=WORKERS,

        # ------------------------------------------------------
        # Mixed precision
        # ------------------------------------------------------
        amp=True,

        # ------------------------------------------------------
        # Pretrained model
        # ------------------------------------------------------
        pretrained=True,

        # ------------------------------------------------------
        # Dataset caching
        # ------------------------------------------------------
        cache=False,

        # ------------------------------------------------------
        # Data augmentation
        # ------------------------------------------------------
        mosaic=1.0,
        close_mosaic=10,

        # ------------------------------------------------------
        # Validation
        # ------------------------------------------------------
        val=True,

        # ------------------------------------------------------
        # Saving
        # ------------------------------------------------------
        save=True,
        save_period=1,

        # ------------------------------------------------------
        # Output directory
        # ------------------------------------------------------
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,

        # ------------------------------------------------------
        # IMPORTANT:
        # Do NOT resume an old experiment
        # ------------------------------------------------------
        resume=False,

        # ------------------------------------------------------
        # Early stopping
        # ------------------------------------------------------
        patience=10,

        # ------------------------------------------------------
        # Reproducibility
        # ------------------------------------------------------
        seed=42,

        # ------------------------------------------------------
        # Generate result plots
        # ------------------------------------------------------
        plots=True,

        # ------------------------------------------------------
        # Logging
        # ------------------------------------------------------
        verbose=True,
    )


    # ==========================================================
    # TRAINING COMPLETE
    # ==========================================================

    elapsed = time.time() - start_time

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 70)
    print("FINAL TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"\nTotal training time: "
        f"{hours}h {minutes}m {seconds}s"
    )

    # ==========================================================
    # RESULT PATHS
    # ==========================================================

    run_directory = os.path.abspath(
        os.path.join(PROJECT, "detect", RUN_NAME)
    )

    best_model = os.path.join(
        run_directory,
        "weights",
        "best.pt"
    )

    last_model = os.path.join(
        run_directory,
        "weights",
        "last.pt"
    )

    print("\nResults directory:")
    print(run_directory)

    print("\nBest model:")
    print(best_model)

    print("\nLast model:")
    print(last_model)

    print("\nImportant result files:")
    print("- results.csv")
    print("- results.png")
    print("- confusion_matrix.png")
    print("- confusion_matrix_normalized.png")
    print("- BoxP_curve.png")
    print("- BoxR_curve.png")
    print("- BoxF1_curve.png")
    print("- BoxPR_curve.png")

    print("\n" + "=" * 70)
    print("TRAINING FINISHED SUCCESSFULLY")
    print("=" * 70)


# ==============================================================
# REQUIRED FOR WINDOWS
# ==============================================================

if __name__ == "__main__":
    main()  