import os
import json

def generate_metadata(image_dir, output_file):
    metadata = []
    
    for category in os.listdir(image_dir):
        category_path = os.path.join(image_dir, category)
        
        if os.path.isdir(category_path):
            for image in os.listdir(category_path):
                image_path = os.path.join(category_path, image)
                
                if os.path.isfile(image_path):
                    # Normalize path to use forward slashes
                    normalized_path = os.path.normpath(image_path).replace(os.sep, '/')
                    formatted_category = category.capitalize()
                    metadata.append({
                        "path": normalized_path,
                        "category": formatted_category
                    })
    
    with open(output_file, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

# Example usage
image_directory = "datasets/freiburg_groceries_dataset/images"
output_file = "datasets/freiburg_groceries_dataset_info.json"
generate_metadata(image_directory, output_file)

print(f"Metadata saved to {output_file}")