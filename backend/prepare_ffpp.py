import os
import random
import shutil
from pathlib import Path

# Paths are relative to DeepfakeDetector folder
RAW_ROOT = Path("../data/ffpp_raw/FF++C32-Frames")
OUT_ROOT = Path("../data/ffpp_processed")

# Folders in RAW_ROOT:
#   Original          -> real
#   Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures -> fake
REAL_FOLDER = "Original"
FAKE_FOLDERS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  # test will implicitly be 0.15


def collect_image_paths():
    real_paths = []
    fake_paths = []

    # Real: all images under Original/*
    real_root = RAW_ROOT / REAL_FOLDER
    for root, _, files in os.walk(real_root):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                real_paths.append(Path(root) / f)

    # Fake: all images under each manipulation folder
    for folder in FAKE_FOLDERS:
        fake_root = RAW_ROOT / folder
        if not fake_root.exists():
            continue
        for root, _, files in os.walk(fake_root):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    fake_paths.append(Path(root) / f)

    print(f"Collected {len(real_paths)} real images and {len(fake_paths)} fake images.")
    return real_paths, fake_paths


def split_paths(paths):
    random.shuffle(paths)
    n = len(paths)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train = paths[:n_train]
    val = paths[n_train:n_train + n_val]
    test = paths[n_train + n_val:]
    return train, val, test


def copy_split(split_name, real_list, fake_list):
    """
    Copies images into:
        OUT_ROOT/split_name/real/*.*
        OUT_ROOT/split_name/fake/*.*
    """
    real_out = OUT_ROOT / split_name / "real"
    fake_out = OUT_ROOT / split_name / "fake"
    real_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)

    def copy_many(src_list, dst_root):
        for src in src_list:
            dst = dst_root / src.name
            # Avoid collisions by prefixing with parent folder if needed
            if dst.exists():
                dst = dst_root / f"{src.parent.name}_{src.name}"
            shutil.copy2(src, dst)

    print(f"Copying {len(real_list)} real images to {real_out}")
    copy_many(real_list, real_out)

    print(f"Copying {len(fake_list)} fake images to {fake_out}")
    copy_many(fake_list, fake_out)


def main():
    random.seed(42)

    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"RAW_ROOT does not exist: {RAW_ROOT.resolve()}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    real_paths, fake_paths = collect_image_paths()

    real_train, real_val, real_test = split_paths(real_paths)
    fake_train, fake_val, fake_test = split_paths(fake_paths)

    print("Real split sizes:", len(real_train), len(real_val), len(real_test))
    print("Fake split sizes:", len(fake_train), len(fake_val), len(fake_test))

    copy_split("train", real_train, fake_train)
    copy_split("val", real_val, fake_val)
    copy_split("test", real_test, fake_test)

    print("Done. Processed data in:", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()
