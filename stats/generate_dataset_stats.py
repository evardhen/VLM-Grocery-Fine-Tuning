import json

def extract_and_safe_dataset_info(dataset_path, output_file):
    # Sample dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Extract properties
    summary = {
        "total_images": len(dataset),  # Count of images
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
    above_5 = 0

    for entry in dataset:
        num_items = len(entry["items"])
        total_items += num_items

        summary["min_items_per_entry"] = min(summary["min_items_per_entry"], num_items)
        summary["max_items_per_entry"] = max(summary["max_items_per_entry"], num_items)
        for item in entry["items"]:
            
            total_count += item["count"]
            if item["count"] > 10:
                above_5 += 1
            summary["min_count_per_entry"] = min(summary["min_count_per_entry"], item["count"])
            summary["max_count_per_entry"] = max(summary["max_count_per_entry"], item["count"])
            
            category = item["category"].lower()
            summary["category_counts"][category] = summary["category_counts"].get(category, 0) + item["count"]
            
            # Collect unique fine-grained categories
            fine_grained = item["fine-grained category"].strip().lower()
            summary["unique_fine_grained_categories"].add(fine_grained)

    summary["average_items_per_entry"] = total_items / len(dataset) if dataset else 0
    summary["average_count_per_entry"] = total_count / len(dataset) if dataset else 0
    summary["category_counts"] = dict(sorted(summary["category_counts"].items(), key=lambda item: item[1], reverse=True))
    # Convert set to list for JSON serialization
    summary["unique_fine_grained_categories"] = list(summary["unique_fine_grained_categories"])

    # Save the summary to a JSON file
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Summary saved to {output_file}")
    print(f"Items above 10 count: {above_5}")

if __name__ == "__main__":
    # dataset_path = "datasets/freiburg_groceries_dataset_info_fixed.json"
    # output_file = "stats/freiburg_dataset_stats.json"
    dataset_path = "datasets/zeki_groceries_dataset_raw.json"
    output_file = "stats/zeki_groceries_dataset_raw_stats.json"
    # dataset_path = "datasets/fruits_and_vegs_dataset_info_preprocessed.json"
    # output_file = "stats/fruits_and_vegs_dataset_info_preprocessed_stats.json"

    extract_and_safe_dataset_info(dataset_path=dataset_path, output_file=output_file)