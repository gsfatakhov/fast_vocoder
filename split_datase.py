import random
from pathlib import Path
import argparse

def split_ljspeech_dataset(dataset_path, train_ratio=0.95, seed=42):
    metadata_path = Path(dataset_path) / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Файл metadata.csv не найден в {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.seed(seed)
    random.shuffle(lines)

    train_size = int(len(lines) * train_ratio)
    train_lines = lines[:train_size]
    val_lines = lines[train_size:]

    train_metadata_path = Path(dataset_path) / "train_metadata.csv"
    val_metadata_path = Path(dataset_path) / "val_metadata.csv"

    with open(train_metadata_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with open(val_metadata_path, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    print(f"Train dataset: {len(train_lines)} lines in {train_metadata_path}")
    print(f"Val dataset: {len(val_lines)} lines in {val_metadata_path}")


if __name__ == "__main__":
    # Парсинг аргументов
    parser = argparse.ArgumentParser(description="Split LJSpeech to train and val.")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to folder LJSpeech-1.1"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.95,
        help="Percentage of data to use for training (default 0.95)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default 42)"
    )
    args = parser.parse_args()

    split_ljspeech_dataset(args.dataset_path, args.train_ratio, args.seed)
