from pathlib import Path

LABEL_DIR = Path("Dataset/RDD_SPLIT/train/labels")

print("=" * 70)
print("SEARCHING FOR INVALID TRAINING LABEL")
print("=" * 70)

found = False

for label_file in LABEL_DIR.glob("*.txt"):

    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            print("\nInvalid number of values:")
            print("File:", label_file)
            print("Line:", line_number)
            print("Content:", line)
            found = True
            continue

        try:
            class_id = int(parts[0])

            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except ValueError:
            print("\nInvalid numeric value:")
            print("File:", label_file)
            print("Line:", line_number)
            print("Content:", line)
            found = True
            continue

        if not (0 <= x <= 1):
            print("\nInvalid X coordinate:")
            print("File:", label_file)
            print("Line:", line_number)
            print("Content:", line)
            found = True

        if not (0 <= y <= 1):
            print("\nInvalid Y coordinate:")
            print("File:", label_file)
            print("Line:", line_number)
            print("Content:", line)
            found = True

        if not (0 < w <= 1):
            print("\nInvalid WIDTH:")
            print("File:", label_file)
            print("Line:", line_number)
            print("Content:", line)
            found = True

        if not (0 < h <= 1):
            print("\nInvalid HEIGHT:")
            print("File:", label_file)
            print("Line:", line_number)
            print("Content:", line)
            found = True


if not found:
    print("\nNo invalid annotations found.")

print("\nSearch completed.")