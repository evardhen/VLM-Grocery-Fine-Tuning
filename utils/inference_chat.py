import json
import csv
import os
from typing import List, Dict, Any
import sys

from llamafactory.chat.chat_model import ChatModel
from llamafactory.extras.misc import torch_gc


class InferenceChat:
    def __init__(self,
                 model_config_path: str,
                 csv_output_path: str, 
                 json_output_path: str, 
                 dataset_path: str,
                 image_resolution: int = 768 * 768):
        self.csv_output_path = csv_output_path
        self.json_output_path = json_output_path
        self.dataset_path = dataset_path
        self.model_config_path = model_config_path
        self.predictions = []
        self.image_resolution = image_resolution

        # Initialize chat model
        sys.argv[1:] = [self.model_config_path]
        self.chat_model = ChatModel()

        # Load dataset
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def run(self, max_dataset_len = None, dataset_start_index = None):
        """
        Run the inference pipeline:
          - Load dataset
          - Perform chat model inference
        """
        self.predictions = []
        total_images = 0
        prompt_token_counts = []
        completion_token_counts = []
        
        dataset_clipped = self.dataset
        if dataset_start_index is not None:
            dataset_clipped = dataset_clipped[dataset_start_index:]
        if max_dataset_len is not None and len(dataset_clipped) > max_dataset_len:
            dataset_clipped = dataset_clipped[:max_dataset_len]
        len_dataset = len(dataset_clipped)


        # Process dataset
        for i, entry in enumerate(dataset_clipped):
            print(f"Processing dataset entry {i}/{len_dataset}")
            messages = [{"role": "user", "content": entry["question"]}]

            # Inference with the chat model
            result = self.chat_model.chat(
                messages=messages,
                images=entry["images"]
            )
            
            # Store prediction
            self.predictions.append({
                "question_id": entry["question_id"],
                "segment_id": entry["segment_id"],
                "question": entry["question"],
                "answer": result[0].response_text,
            })
            
            # Optional: Clear memory after each entry
            torch_gc()

            total_images += len(entry["images"])
            prompt_token_counts.append(result[0].prompt_length)
            completion_token_counts.append(result[0].response_length)

        # Compute statistics
        avg_images_per_query = total_images / len_dataset if len_dataset > 0 else 0
        avg_prompt_tokens_per_query = sum(prompt_token_counts) / len_dataset if len_dataset > 0 else 0
        avg_completion_tokens_per_query = sum(completion_token_counts) / len_dataset if len_dataset > 0 else 0
        max_prompt_tokens_per_query = max(prompt_token_counts)
        min_prompt_tokens_per_query = min(prompt_token_counts)
        max_completion_tokens_per_query = max(completion_token_counts)
        min_completion_tokens_per_query = min(completion_token_counts)

        return {
            "avg_images_per_query": avg_images_per_query,
            "avg_prompt_tokens_per_query": avg_prompt_tokens_per_query,
            "avg_completion_tokens_per_query": avg_completion_tokens_per_query,
            "max_prompt_tokens_per_query": max_prompt_tokens_per_query,
            "min_prompt_tokens_per_query": min_prompt_tokens_per_query,
            "max_completion_tokens_per_query": max_completion_tokens_per_query,
            "min_completion_tokens_per_query": min_completion_tokens_per_query
        }
    
    def chat(self, query: str, images = []):
        """
        query: Text input prompt
        images: List of image paths
        """

        # Process dataset
        print(f"\nDetected {len(images)} images.")
        print(f"Processing query...")
        messages = [{"role": "user", "content": query}]

        # Inference with the chat model
        result = self.chat_model.chat(
            messages=messages,
            images=images
        )
        print("Got result from VLM...")
        
        return result[0].response_text
        
    def clear_chat_history():
        # Optional: Clear memory after each entry
        torch_gc()

    def save_json(self) -> None:
        try:
            with open(self.json_output_path, "w", encoding="utf-8") as f:
                json.dump(self.predictions, f, indent=4)
            print(f"Predictions saved to {self.json_output_path}")
        except Exception as e:
            print(f"Failed to save JSON. Error: {e}")

    def save_csv(self) -> None:
        try:
            # In case the list is empty or has different keys, handle generically
            if not self.predictions:
                print("No data to write to CSV.")
                return
            
            # Use the keys from the first dictionary as CSV headers
            fieldnames = list(self.predictions[0].keys())

            with open(self.csv_output_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.predictions)

            print(f"Predictions saved to {self.csv_output_path}")
        except Exception as e:
            print(f"Failed to save CSV. Error: {e}")
