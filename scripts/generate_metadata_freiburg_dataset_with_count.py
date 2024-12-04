import os
import json
import re

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers.openai_api import (send_openai_request)


def analyze_image(image_path, category):
    try:

        message_content = send_openai_request(image_path=image_path, category=category)
        print(message_content)
        return extract_count(message_content), extract_fine_grained_category(message_content)

    except Exception as e:
        with open("error_log_openai.txt", "a") as file:
            print(f"Error analyzing image {image_path}: {e}")
            file.write(f"Error analyzing image {image_path}: {e}")
        return "NA"

def extract_count(description):
    words = description.split()
    for i, word in enumerate(words):
        if word.isdigit():  # Look for a number
            return int(word)
    return "NA"  # Default if no count found

def extract_fine_grained_category(prompt):
    match = re.search(r"category:\s*(.+)", prompt, re.IGNORECASE)
    if match:
        return match.group(1).strip()  # Extract the text after 'Fine-grained category:'
    return "NA"  # Default if no match is found

def generate_metadata_with_counts(image_dir):
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
                    count, fine_category = analyze_image(normalized_path, formatted_category)
                    
                    metadata.append({
                        "path": normalized_path,
                        "category": formatted_category,
                        "fine-grained category": fine_category,
                        "count": count
                    })

                    print(f"{i}\\5000")
                    i += 1

    return metadata

def save_metadata(output_file, metadata):
    with open(output_file, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

# Example usage
image_directory = "datasets/freiburg_groceries_dataset/images"
output_json = "datasets/freiburg_groceries_dataset_info_with_counts.json"
metadata = generate_metadata_with_counts(image_directory)
save_metadata(output_json, metadata)
print(f"Metadata with counts saved to {output_json}")
