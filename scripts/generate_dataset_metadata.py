import os
import json
import re
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.openai_api import (send_openai_request)

logging.basicConfig(
    filename="logs/metadata_logs.log",  # Specify the log file name
    level=logging.WARNING,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format for log messages
    filemode="a",  # Append mode to preserve existing logs
)

def analyze_image(image_path, dataset, prompt_path, category = None):
    try:
        message_content = send_openai_request(image_path=image_path, dataset=dataset, category=category, prompt_path=prompt_path)
        print(message_content)
        return extract_info_from_string(message_content, dataset)

    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")
        logging.error(f"Error analyzing image {image_path}: {e}\n")
        return "NA"
    

def extract_info_from_string(text, dataset):
    if dataset == "zeki":
        pattern = r"Category:\s*(.+?)\s*Count:\s*(\d+)\s*Fine-grained category:\s*(.+)"
    elif dataset == "freiburg":
        pattern = r"Category:\s*(.+?)\s*Coarse-category:\s*(.+?)\s*Count:\s*(\d+)\s*Fine-grained category:\s*(.+)"
    elif dataset == "fruits_and_vegs":
        pattern = r"Category:\s*(.+?)\s*Coarse-category:\s*(.+?)\s*Count:\s*(\d+)"
    else:
        raise NotImplementedError

    # Use re.findall to extract all matches
    matches = re.findall(pattern, text, re.IGNORECASE)

    # Convert the matches into a list of dictionaries
    if dataset == "freiburg":
        results = [
            {"category": match[1], "count": int(match[2]), "fine-grained category": match[3]}
            for match in matches
        ]
    elif dataset == "zeki":
        results = [
            {"category": match[0], "count": int(match[1]), "fine-grained category": match[2]}
            for match in matches
        ]
    elif dataset == "fruits_and_vegs":
        results = [
            {"category": match[1], "count": int(match[2]), "fine-grained category": match[0]}
            for match in matches
        ]       
    if not results:
        logging.info("No matches found in the provided text.")
    return results

def generate_dataset_metadata(image_dir, dataset_name, prompt_path):
    metadata = []
    i = 0
    for category in os.listdir(image_dir):
        category_path = os.path.join(image_dir, category)
        if os.path.isdir(category_path):
            for image in os.listdir(category_path):
                image_path = os.path.join(category_path, image)
                if os.path.isfile(image_path):
                    normalized_path = os.path.normpath(image_path).replace(os.sep, '/')
                    formatted_category = category.capitalize()
                    
                    # Get count using OpenAI API
                    results = analyze_image(normalized_path, dataset=dataset_name, category=formatted_category, prompt_path=prompt_path)
                    metadata.append({
                        "path": normalized_path,
                        "items": results
                    })

                    print(f"{i}\\3300")
                    i += 1

    return metadata

def generate_zeki_groceries_metadata(image_dir):
    metadata = []
    i = 0
    if not os.path.isdir(image_dir):
        raise FileNotFoundError
    
    for image in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image)
        if os.path.isfile(image_path):
            normalized_path = os.path.normpath(image_path).replace(os.sep, '/')
            
            # Get count using OpenAI API
            results = analyze_image(normalized_path, dataset="zeki", prompt_path="configs/prompts/zeki_groceries_dataset_prompts.yaml")
            
            metadata.append({
                "path": normalized_path,
                "items": results
            })

            print(f"{i}\\500")
            i += 1

    return metadata

def save_metadata(output_file, metadata):
    with open(output_file, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)
        

def main():
    # # Usage for Freiburg Dataset
    # freiburg_image_directory = "datasets/freiburg_groceries_dataset/images"
    # freiburg_output_json = "datasets/freiburg_groceries_dataset_info_raw_v2.json"
    # metadata = generate_dataset_metadata(freiburg_image_directory, dataset_name="freiburg", prompt_path="configs/prompts/freiburg_groceries_dataset_prompts.yaml")
    # save_metadata(freiburg_output_json, metadata)
    # print(f"Metadata with counts saved to {freiburg_output_json}")

    # Usage for Zeki Dataset
    freiburg_image_directory = "datasets/binaries/captured_images_home"
    freiburg_output_json = "datasets/zeki_2_groceries_dataset_info.json"
    metadata = generate_zeki_groceries_metadata(freiburg_image_directory)
    save_metadata(freiburg_output_json, metadata)
    print(f"Metadata with counts saved to {freiburg_output_json}")

    # # Usage for Fruits and Vegs Dataset
    # fruits_and_vegs_image_directory = "datasets/fruits_and_vegs"
    # fruits_and_vegs_output_json = "datasets/fruits_and_vegs_dataset_info.json"
    # metadata = generate_dataset_metadata(fruits_and_vegs_image_directory, dataset_name="fruits_and_vegs", prompt_path="configs/prompts/fruits_and_vegs_dataset_prompts.yaml")
    # save_metadata(fruits_and_vegs_output_json, metadata)
    # print(f"Metadata with counts saved to {fruits_and_vegs_output_json}")

if __name__ == "__main__":
    main()