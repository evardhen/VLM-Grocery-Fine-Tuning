import json
from PIL import Image

def _get_image_resolution(image_path):
    """
    Open the image and return its resolution as (area, (width, height)).
    Area is computed as width * height.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width * height
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
    
def extract_and_safe_dataset_info(dataset_path, dataset_path_alpaca_format, output_file):
    # Sample dataset
    dataset = []
    for path in dataset_path:
        with open(path, "r") as f:
            dataset += json.load(f)

    with open(dataset_path_alpaca_format, "r") as f:
        dataset_alpaca = json.load(f)

    total_entries = len(dataset)
    question_lengths = [len(entry["question"].split()) for entry in dataset_alpaca]
    answer_lengths = [len(entry["answer"].split()) for entry in dataset_alpaca]
    # Extract properties
    summary = {
        "avg_images_per_query": 1,
        "max_images_per_query": 1,
        "min_images_per_query": 1,
        "avg_question_length": sum(question_lengths) / total_entries if total_entries > 0 else 0,
        "max_question_length": max(question_lengths, default=0),
        "avg_answer_length": sum(answer_lengths) / total_entries if total_entries > 0 else 0,
        "max_answer_length": max(answer_lengths, default=0),
        "average_resolution": 0,
        "min_resolution": float('inf'),
        "max_resolution": float('-inf'),
        "total_images": total_entries,  # Count of images
        "average_items_per_entry": 0,
        "min_items_per_entry": float('inf'),
        "max_items_per_entry": float('-inf'),
        "average_count_per_entry": 0,
        "min_count_per_entry": float('inf'),
        "max_count_per_entry": float('-inf'),
        "category_counts": {},  # Total count of items per category
        "unique_fine_grained_categories": set()  # Unique fine-grained categories
    }

    total_items = 0
    total_count = 0
    total_res = 0

    for entry in dataset:
        res = _get_image_resolution(entry["path"])
        summary["min_resolution"] = min(summary["min_resolution"], res)
        summary["max_resolution"] = max(summary["max_resolution"], res)
        total_res += res

        num_items = len(entry["items"])
        total_items += num_items

        summary["min_items_per_entry"] = min(summary["min_items_per_entry"], num_items)
        summary["max_items_per_entry"] = max(summary["max_items_per_entry"], num_items)
        for item in entry["items"]:
            
            total_count += item["count"]
            summary["min_count_per_entry"] = min(summary["min_count_per_entry"], item["count"])
            summary["max_count_per_entry"] = max(summary["max_count_per_entry"], item["count"])
            
            category = item["category"].lower()
            summary["category_counts"][category] = summary["category_counts"].get(category, 0) + item["count"]
            
            # Collect unique fine-grained categories
            fine_grained = item["fine-grained category"].strip().lower()
            summary["unique_fine_grained_categories"].add(fine_grained)

    summary["average_resolution"] = total_res / len(dataset) if dataset else 0
    summary["average_items_per_entry"] = total_items / len(dataset) if dataset else 0
    summary["average_count_per_entry"] = total_count / len(dataset) if dataset else 0
    summary["category_counts"] = dict(sorted(summary["category_counts"].items(), key=lambda item: item[1], reverse=True))
    # Convert set to list for JSON serialization
    summary["unique_fine_grained_categories"] = list(summary["unique_fine_grained_categories"])

    # Save the summary to a JSON file
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Summary saved to {output_file}")

if __name__ == "__main__":
    # dataset_path = "data/metadata/final/freiburg_grocery_dataset.json"
    # dataset_path_alpaca = "data/freiburg_grocery.json"
    # output_file = "stats/dataset_stats/freiburg_dataset_final_stats.json"
    # dataset_path = "data/metadata/final/fruits_and_vegs_dataset.json"
    # dataset_path_alpaca = "data/fruits_and_vegs_grocery.json"
    # output_file = "stats/dataset_stats/fruits_and_vegs_dataset_final_stats.json"
    # extract_and_safe_dataset_info(dataset_path=[dataset_path], dataset_path_alpaca_format=dataset_path_alpaca, output_file=output_file)


    dataset_path = "data/metadata/final/zeki_grocery_dataset.json"
    dataset_path_2 = "data/metadata/final/zeki_2_grocery_dataset.json"
    dataset_path_alpaca = "data/zeki_grocery.json"
    output_file = "stats/dataset_stats/zeki_grocery_dataset_final_stats.json"
    extract_and_safe_dataset_info(dataset_path=[dataset_path, dataset_path_2], dataset_path_alpaca_format=dataset_path_alpaca, output_file=output_file)