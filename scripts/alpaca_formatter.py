import json
import yaml

BASE_QUESTION = "<image>List the groceries in the image alongside their categories, fine-grained categories, and counts."

def alpaca_format(path_list, system_prompt):
    dataset = []
    for dataset_path in path_list:
        with open(dataset_path, "r") as f:
            dataset.extend(json.load(f))

    formatted_dataset = []
    print(f"Dataset length: {len(dataset)}")

    for entry in dataset:
        new_entry = {
            "question": BASE_QUESTION,
            "answer": None,
            "images": [entry["path"]],
            "system": system_prompt,
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
    with open("configs/prompts/system_prompts.yaml", "r") as file:
        data = yaml.safe_load(file)
    # SYSTEM_PROMPT_SIMPLE = data["simple_prompt"]
    SYSTEM_PROMPT_FEW_SHOT = data["few_shot_prompt"]
    # SYSTEM_PROMPT_JSON_SIMPLE = data["json_simple_prompt"]
    # SYSTEM_PROMPT_JSON_FEW_SHOT = data["json_few_shot_prompt"]
    # dataset_path = "data/metadata/final/freiburg_grocery_dataset.json"
    # dataset_path = "data/metadata/final/fruits_and_vegs_dataset.json"
    dataset_path = "data/predictions/openai/o3_raw.json"
    # dataset_path_2 = "data/predictions/openai/zeki_2_groceries_dataset_raw.json"
    output_path_simple = "data/predictions/openai/openai_o3_predictions.json"
    # output_path_few_shot = "data/zeki_grocery_few_shot_prompt.json"
    # output_path_simple_json = "data/zeki_grocery_simple_json_prompt.json"
    # output_path_few_shot_json = "data/zeki_grocery_few_shot_json_prompt.json"
    
    path_list = [dataset_path]
    data = alpaca_format(path_list, SYSTEM_PROMPT_FEW_SHOT)
    save_alpaca_format(data, output_path_simple)
    # data = alpaca_format(path_list, SYSTEM_PROMPT_FEW_SHOT)
    # save_alpaca_format(data, output_path_few_shot)
    # data = alpaca_format(path_list, SYSTEM_PROMPT_FEW_SHOT)
    # save_alpaca_format(data, output_path_simple_json)
    # data = alpaca_format(path_list, SYSTEM_PROMPT_FEW_SHOT)
    # save_alpaca_format(data, output_path_few_shot_json)