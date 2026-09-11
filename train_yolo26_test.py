from ultralytics import YOLO
import torch
import os


def main():

    print("=" * 70)
    print("YOLO26n RDD2022 FINAL TRAINING")
    print("=" * 70)

    # ---------------------------------------------------------
    # GPU CHECK
    # ---------------------------------------------------------
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Training should NOT be started."
        )

    print("GPU:", torch.cuda.get_device_name(0))

    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"VRAM: {vram:.1f} GB")

    # ---------------------------------------------------------
    # PATHS
    # ---------------------------------------------------------
    DATASET = os.path.abspath("./data.yaml")

    if not os.path.exists(DATASET):
        raise FileNotFoundError(
            f"Dataset configuration not found:\n{DATASET}"
        )

    print("Dataset:", DATASET)

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------
    print("\nLoading YOLO26n...")

    model = YOLO("yolo26n.pt")

    print("YOLO26n loaded successfully.")

    # ---------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------
    print("\nStarting training...")
    print("=" * 70)

    results = model.train(

        # Dataset
        data=DATASET,

        # Model training
        epochs=1,
        imgsz=640,
        batch=24,

        # GPU
        device=0,

        # Windows-safe dataloader
        workers=0,

        # Mixed precision
        amp=True,

        # Pretrained weights
        pretrained=True,

        # Dataset caching
        cache=False,

        # Augmentation / training
        mosaic=1.0,
        close_mosaic=10,

        # Validation
        val=True,

        # Save results
        save=True,
        save_period=1,

        # Output
        project="./runs",
        name="yolo26n_final",

        # Don't accidentally resume old training
        resume=False,
        exist_ok=True,

        # Early stopping
        patience=5,

        # Reproducibility
        seed=42,

        # Generate plots
        plots=True,

        # Display progress
        verbose=True
    )

    # ---------------------------------------------------------
    # COMPLETED
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("YOLO26n TRAINING COMPLETED")
    print("=" * 70)

    print("\nResults saved to:")
    print("./runs/detect/yolo26n_final")

    print("\nBest model:")
    print("./runs/detect/yolo26n_final/weights/best.pt")

    print("\nLast model:")
    print("./runs/detect/yolo26n_final/weights/last.pt")


if __name__ == "__main__":
    main()