import json
import re
from PIL import Image

def remove_empty_items(dataset_path):
    new_data = []
    with open(dataset_path, "r") as f:
        data = json.load(f)

    for entry in data:
        if entry["items"] == []:
            continue
        new_data.append(entry)
    return new_data

def remove_item_count_zero(dataset_path):
    new_data = []
    item_count = 0
    entry_count = 0

    with open(dataset_path, "r") as f:
        data = json.load(f)

    for entry in data:
        new_entry = {"path": entry["path"], "items": []}
        for item in entry["items"]:
            if item["count"] == 0:
                item_count += 1
                continue
            new_entry["items"].append(item)

        if new_entry["items"] != []:
            new_data.append(new_entry)
            continue
        entry_count += 1

    print(f"Removed {item_count} dataset items of {len(data)}.")
    print(f"Removed {entry_count} dataset entries of {len(data)}.")
    return new_data

def fix_wrong_category_allocation_vegs_and_fruits(dataset_path):
    item_count = 0

    with open(dataset_path, "r") as f:
        data = json.load(f)

    for entry in data:
        for item in entry["items"]:
            if item["category"].lower().strip() not in ["vegetable", "fruit"] and item["fine-grained category"] in ["vegetable", "fruit"]:
                tmp = item["category"]
                item["category"] = item["fine-grained category"]
                item["fine-grained category"] = tmp
                item_count += 1
                

    print(f"Reordered {item_count} dataset items of {len(data)}.")
    return data

def remove_items_count_above(dataset_path, threshold):
    new_data = []
    count = 0
    skip_entry = False

    with open(dataset_path, "r") as f:
        data = json.load(f)

    for entry in data:
        for item in entry["items"]:
            if item["count"] > threshold:
                count += 1
                skip_entry = True
                continue

        if not skip_entry:
            new_data.append(entry)
        skip_entry = False
    print(f"Removed {count} dataset entries of {len(data)}.")
    return new_data

def remove_items_per_image_above(dataset_path, threshold):
    new_data = []
    count = 0

    with open(dataset_path, "r") as f:
        data = json.load(f)

    for entry in data:
        if len(entry["items"]) > threshold:
            count += 1
            continue
        new_data.append(entry)

    print(f"Removed {count} dataset entries of {len(data)}.")
    return new_data

def remove_brackets_from_items(dataset_path):

    with open(dataset_path, "r") as f:
        data = json.load(f)

    for entry in data:
        for item in entry["items"]:
            item["fine-grained category"] = re.sub(r'\s*\(.*?\)\s*', '', item["fine-grained category"]).strip()

    return data

def remove_resolution_above(dataset_path, threshold):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    filtered_dataset = []
    count = 0
    for entry in dataset:
        try:
            with Image.open(entry["path"]) as img:
                width, height = img.size
                resolution = width * height

            if resolution <= threshold:
                # Optionally update the path
                filtered_dataset.append(entry)
            else:
                count += 1

        except Exception as e:
            print(f"Skipping {entry['path']}: {e}")

    print(f"{count} items above the threshold of: {threshold}")
    return filtered_dataset


def save_dataset(output_file, data):
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def main():
    dataset_path = "data/metadata/preprocessed/zeki_2_groceries_dataset.json"
    output_path = "data/metadata/final/zeki_2_grocery_dataset.json"

    # new_data = remove_empty_items(dataset_path)
    # new_data = remove_items_count_above(dataset_path, threshold=8)
    # new_data = remove_items_per_image_above(dataset_path, threshold=3)
    # new_data = remove_item_count_zero(dataset_path)
    # new_data = fix_wrong_category_allocation_vegs_and_fruits(dataset_path)
    # new_data = remove_brackets_from_items(dataset_path)
    new_data = remove_resolution_above(dataset_path, 10000000)

    save_dataset(output_path, new_data)

if __name__ == "__main__":
    main()
