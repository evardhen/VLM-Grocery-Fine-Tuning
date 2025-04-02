import json

BASE_QUESTION = "<image>List the groceries in the image alongside their categories, fine-grained categories, and counts."
SYSTEM_PROMPT = "You are grocery detector who scans the groceries in an image. For each item, provide the fine-grained category, the coarse category and the count. Per fine-grained category, list how many there are. If you cannot identify a specific fine-grained category, repeat the coarse category for both. If multiple items of the same category but different fine-grained category appear (e.g., different variations, brands), list each separately."

def alpaca_format(dataset_path):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    formatted_dataset = []

    for entry in dataset:
        new_entry = {
            "question": BASE_QUESTION,
            "answer": None,
            "images": [entry["path"]],
            "system": SYSTEM_PROMPT,
            }
        answer = "" if len(entry["items"]) != 0 else "There are no items in the image."

        for item in entry["items"]:
            answer += f"Category: {item['category']}\n"
            answer += f"Fine-grained category: {item['fine-grained category']}\n"
            answer += f"Count: {item['count']}\n\n"
        
        new_entry["answer"] = answer
        formatted_dataset.append(new_entry)
    
    return formatted_dataset

def save_alpaca_format(dataset, path):
    with open(path, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"File saved successfully to: {path}")

if __name__ == "__main__":
    dataset_path = "datasets/metadata/final/freiburg_groceries_dataset_final.json"
    output_path = "datasets/freiburg_groceries.json"
    data = alpaca_format(dataset_path)
    save_alpaca_format(data, output_path)