import json
import random
import re
from pathlib import Path

SEED = 42
RE_IMAGE = re.compile(r"captured_image_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def collect_images(folder: Path) -> list[tuple[int, Path]]:
    """Return ``[(index, path), ...]`` for every image matching the pattern."""

    images: list[tuple[int, Path]] = []
    folder = Path(folder)
    for path in folder.glob("captured_image_*"):
        if m := RE_IMAGE.match(path.name):
            images.append((int(m.group(1)), path))
        else:
            print("Image did not match pattern!!!")
    return images

def main():

    random.seed(SEED)

    captured = collect_images("data/binaries/captured_images")
    captured_home = collect_images("data/binaries/captured_images_home")

    # ----- fixed eval subsets ------------------------------------------------
    eval_paths: set[Path] = set()

    eval_paths.update(path for idx, path in captured if 440 <= idx <= 502)
    eval_paths.update(path for idx, path in captured_home if 272 <= idx <= 325)

    # ----- extra random eval images -----------------------------------------
    all_images = [path for _, path in captured + captured_home]
    remaining = [p for p in all_images if p not in eval_paths]
    eval_paths.update(random.sample(remaining, 89))

    # ----- build train / eval entries ---------------------------------------
    train_paths = [p for p in all_images if p not in eval_paths]

    with open("data/zeki_grocery.json", "r") as f:
        data = json.load(f)
    
    eval_entries = []
    train_entries = []
    for entry in data:
        img_path = Path(entry["images"][0])   
        if img_path in eval_paths:
            eval_entries.append(entry)
        elif img_path in train_paths:
            train_entries.append(entry)

    print(f"Zeki dataset size: {len(data)}")
    print(f"Wrote {len(train_entries)} training samples.")
    print(f"Wrote {len(eval_entries)} evaluation samples")

    with open("data/zeki_grocery_train.json", "w") as json_file:
        json.dump(train_entries, json_file, indent=4)
    with open("data/eval.json", "w") as json_file:
        json.dump(eval_entries, json_file, indent=4)

if __name__ == "__main__":
    main()
