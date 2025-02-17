import json
import re

def fix_metadata(input_json_path, output_json_path):
    """
    Reads a JSON file of image metadata, detects malformed entries
    (where multiple items are concatenated into one string), and fixes them
    by splitting them into multiple items.
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fixed_data = []
    skipped_entries = 0

    for entry in data:
        # Each entry is something like:
        # {
        #   "path": "...",
        #   "items": [
        #       {
        #         "category": "...",
        #         "count": ...,
        #         "fine-grained category": "..."
        #       }
        #   ]
        # }
        new_items = []
        for item in entry["items"]:
            # Check if there's a newline that suggests multiple items got concatenated
            fg_category = item.get("fine-grained category", "")
            if "\n" not in fg_category:
                # It's a normal item, just add it
                new_items.append(item)
            else:
                # We have multiple items mashed into the 'fine-grained category' field.
                # Split them on double-newline or blank lines:
                # e.g. "Tarte Chocolat\n\nCategory: Donau-Wellen\nCoarse-category: Sweets\nCount: 1\n..."
                # becomes separate chunks for each item.
                chunks = re.split(r"\n\s*\n", fg_category.strip())
                prev_item = chunks[0]
                chunks = chunks[1:]
                
                for chunk in chunks:
                    # Initialize item dict using fallback from original
                    chunk_item = {
                        "category": "NA",
                        "count": 1,
                        "fine-grained category": "NA"
                    }
                    
                    # If the chunk is just a single line (like "Tarte Chocolat"),
                    # we interpret it as the 'fine-grained category'.
                    # If the chunk has multiple lines like
                    #   Category: Donau-Wellen
                    #   Coarse-category: Sweets
                    #   ...
                    # we parse them individually.
                    
                    lines = chunk.splitlines()
                    # We have multiple lines to parse
                    for line in lines:
                        line = line.strip()
                        # Example lines: "Category: Donau-Wellen", "Coarse-category: Sweets", ...
                        if not line:
                            continue
                        parts = line.split(':', 1)
                        if len(parts) != 2:
                            skipped_entries += 1
                            item["fine-grained category"] = prev_item
                            new_items.append(item)
                            continue
                        
                        key, val = parts
                        key = key.strip().lower()
                        val = val.strip()
                        
                        # Map keys we see in the chunk to our item fields:
                        if key == "coarse-category":
                            chunk_item["category"] = val
                        elif key == "fine-grained category":
                            chunk_item["fine-grained category"] = val
                        elif key == "count":
                            # Convert to integer if possible
                            try:
                                chunk_item["count"] = int(val)
                            except ValueError:
                                chunk_item["count"] = val
                                print("Not a number...")
                        elif key == "category":
                            pass
                        else:
                            print("Not implemented")
                    
                    # Now we have a properly formed item
                    new_items.append(chunk_item)
        
        # Replace the old items list with our new items
        entry["items"] = new_items
        fixed_data.append(entry)

    # Write out the corrected JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=4, ensure_ascii=False)

    print(f"Skipped: {skipped_entries}")


if __name__ == "__main__":
    # Example usage:
    input_file = "datasets/freiburg_groceries_dataset_info_updated.json"
    output_file = "datasets/freiburg_groceries_dataset_info_fixed.json"
    fix_metadata(input_file, output_file)
    print(f"Fixed metadata written to {output_file}")
