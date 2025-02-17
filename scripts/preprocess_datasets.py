import json
import re

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

def save_dataset(output_file, data):
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def main():
    dataset_path = "datasets/freiburg_groceries_dataset_info_preprocessed_v2.json"
    output_path = "datasets/freiburg_groceries_dataset_info_preprocessed_v2.json"

    # dataset_path = "datasets/fruits_and_vegs_dataset_info_preprocessed.json"
    # output_path = "datasets/fruits_and_vegs_dataset_info_preprocessed.json"

    # new_data = remove_empty_items(dataset_path)
    # new_data = remove_items_count_above(dataset_path, threshold=8)
    # new_data = remove_items_per_image_above(dataset_path, threshold=3)
    # new_data = remove_item_count_zero(dataset_path)
    # new_data = fix_wrong_category_allocation_vegs_and_fruits(dataset_path)
    new_data = remove_brackets_from_items(dataset_path)

    save_dataset(output_path, new_data)

if __name__ == "__main__":
    main()
