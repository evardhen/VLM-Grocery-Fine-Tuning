import os
import yaml
import json
import csv
import requests
import concurrent.futures

from openai import OpenAI
from transformers.utils.versions import require_version

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")


def assert_url_accessible(url: str, timeout: float = 5.0) -> None:
    """
    Checks if the URL is accessible (HTTP status code 200–399).
    Raises an exception if it isn't accessible or if any request error occurs.
    
    :param url: The URL to check.
    :param timeout: How many seconds to wait for a response before timing out.
    :raises ValueError: If the URL is not accessible.
    """
    try:
        response = requests.head(url, timeout=timeout)
        if not response.ok:
            raise ValueError(f"URL returned status code {response.status_code}: {url}")
    except requests.RequestException as e:
        raise ValueError(
            f"URL is not accessible. Start a local http server as described in the README: {url}, reason: {e}"
        )


class InferenceAPI:
    """
    A class to load configs/dataset, check local image URLs, call the LlamaFactory endpoint,
    and save predictions to JSON/CSV.
    """

    def __init__(self, model_config_path, dataset_path, json_output_path, csv_output_path):
        self.model_config_path = model_config_path
        self.dataset_path = dataset_path
        self.json_output_path = json_output_path
        self.csv_output_path = csv_output_path
        self.predictions = []

        # 1) Load model config
        with open(self.model_config_path, "r") as file:
            configs = yaml.safe_load(file)
            self.model_name_or_path = configs["model_name_or_path"]

        # 2) Load dataset
        with open(self.dataset_path, "r") as file:
            self.dataset = json.load(file)

        # 3) Check if HTTP image server is available for the first item
        example_url = self.dataset[0]["images"][0]
        test_url = f"http://localhost:8001/{example_url}"
        assert_url_accessible(test_url)

        # 4) Configure the local LlamaFactory (OpenAI‑style) endpoint
        self.client = OpenAI(
            api_key=os.environ.get("API_KEY", "0"),
            base_url="http://{}:{}/v1".format(
                os.environ.get("API_HOST", "0.0.0.0"),
                os.environ.get("API_PORT", "8080"),
            ),
        )

    def run(self, max_dataset_len=None, dataset_start_index=None, num_threads=1):
        """
        Run the prediction loop on self.dataset, in parallel using threads, 
        and return usage statistics.
        """
        dataset_clipped = self.dataset
        if dataset_start_index is not None:
            dataset_clipped = dataset_clipped[dataset_start_index:]
        if max_dataset_len is not None and len(dataset_clipped) > max_dataset_len:
            dataset_clipped = dataset_clipped[:max_dataset_len]

        len_dataset = len(dataset_clipped)
        print(f"\nProcessing a dataset of size {len_dataset}\n")
        
        self.predictions = []
        prompt_token_counts = []
        completion_token_counts = []
        total_images = 0
        index = 0
        
        def _process_entry(entry):
            """
            Process a single dataset entry: 
            1. Build messages
            2. Call the API
            3. Return the prediction and token counts
            """

            # Prepare content with question + images
            content = [{"type": "text", "text": entry["question"]}]
            for image_path in entry["images"]:
                url = f"http://localhost:8001/{image_path}"
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    }
                )

            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]

            result = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name_or_path
            )

            prediction = {
                "question_id": entry["question_id"],
                "segment_id": entry["segment_id"],
                "question": entry["question"],
                "answer": result.choices[0].message.content,
            }

            return (prediction, result.usage.prompt_tokens, result.usage.completion_tokens, len(entry["images"]))
        
        # Use ThreadPoolExecutor to run inference in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all entries to the thread pool
            futures = [
                executor.submit(_process_entry, entry)
                for entry in dataset_clipped
            ]

            # As each job completes, gather its results
            for future in concurrent.futures.as_completed(futures):
                prediction, prompt_tokens, completion_tokens, n_images = future.result()
                print(f"Processed dataset entry {index}/{len_dataset}")
                index += 1

                self.predictions.append(prediction)
                prompt_token_counts.append(prompt_tokens)
                completion_token_counts.append(completion_tokens)
                total_images += n_images

        # Compute statistics
        avg_images_per_query = total_images / len_dataset if len_dataset > 0 else 0
        avg_prompt_tokens_per_query = sum(prompt_token_counts) / len_dataset if len_dataset > 0 else 0
        avg_completion_tokens_per_query = sum(completion_token_counts) / len_dataset if len_dataset > 0 else 0
        max_prompt_tokens_per_query = max(prompt_token_counts) if len_dataset > 0 else 0
        min_prompt_tokens_per_query = min(prompt_token_counts) if len_dataset > 0 else 0
        max_completion_tokens_per_query = max(completion_token_counts) if len_dataset > 0 else 0
        min_completion_tokens_per_query = min(completion_token_counts) if len_dataset > 0 else 0

        return {
            "avg_images_per_query": avg_images_per_query,
            "avg_prompt_tokens_per_query": avg_prompt_tokens_per_query,
            "avg_completion_tokens_per_query": avg_completion_tokens_per_query,
            "max_prompt_tokens_per_query": max_prompt_tokens_per_query,
            "min_prompt_tokens_per_query": min_prompt_tokens_per_query,
            "max_completion_tokens_per_query": max_completion_tokens_per_query,
            "min_completion_tokens_per_query": min_completion_tokens_per_query
        }

    def save_json(self):
        # Save JSON
        with open(self.json_output_path, "w", encoding="utf-8") as file:
            json.dump(self.predictions, file, indent=4, ensure_ascii=False)
        print(f"JSON file '{self.json_output_path}' created successfully!")

    def save_csv(self):
        # Save CSV
        with open(self.csv_output_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["question_id", "segment_id", "question", "answer"])
            writer.writeheader()
            writer.writerows(self.predictions)
        print(f"CSV file '{self.csv_output_path}' created successfully.")
