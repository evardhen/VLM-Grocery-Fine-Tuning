import time
import json
import os
import traceback, sys, multiprocessing as mp

from utils.inference_vllm import InferenceVLLM
from utils.inference_api import InferenceAPI
from utils.inference_chat import InferenceChat
from utils.inference_huggingface import InferenceHuggingface

def compute_and_safe_inference_statistics(file_path, type="api", query = "single", images = "single", num_threads = None, batch_size=8):
    """
    Computing statistics on 500 samples of each dataset.
    Parameters:
    -----------
    file_path : str
    type : str, Possible values:
        - "api"  : Uses an API-based inference method.
        - "chat" : Uses a chat-based inference model.
        - "vllm" : Uses a vLLM (optimized inference framework).
    query : str, Possible values:
        - "single" : Each inference call processes a single query.
        - "multiple" : Each inference call processes multiple queries.
    images : str, Possible values:
        - "single" : Each query is associated with a single image.
        - "multiple" : Each query can have multiple images.
    """
    DATASET_LEN = 200


    # Set dataset path
    if images == "multiple":
        dataset_path = "data/eval_dataset_three_images.json"
        print("\nUsing dataset with multiple images per entry...")
    elif images == "single":
        dataset_path = "data/eval_dataset_single_image.json"
        print("\nUsing dataset with single image per entry...")
    else:
        raise NotImplementedError
    
    # Select inference type
    if type == "api":
        inference_pipeline = InferenceAPI(
            model_config_path="configs/inference_configs/adapted_inference_image_resolution/qwen2vl_base.yaml",
            dataset_path=dataset_path,
            json_output_path="none",
        )
    elif type == "chat":
        inference_pipeline = InferenceChat(
            model_config_path="configs/inference_configs/adapted_inference_image_resolution/qwen2vl_base.yaml",
            dataset_path=dataset_path,
            json_output_path="none",
        )
    elif type == "vllm":
        inference_pipeline = InferenceVLLM(
            model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
            dataset=os.path.splitext(os.path.basename(dataset_path))[0],
            template="qwen2_vl",
            json_output_path="none",
            batch_size=batch_size,
            adapter_name_or_path="saves/qwen2.5_vl-7b/lora/sft/grocery_589824_res"
            )
    elif type == "huggingface":
        inference_pipeline = InferenceHuggingface(
            model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
            dataset=os.path.splitext(os.path.basename(dataset_path))[0],
            template="qwen2_vl",
            json_output_path="none",
            batch_size=batch_size,
            adapter_name_or_path="saves/qwen2.5_vl-7b/lora/sft/grocery_589824_res",
            )
    else:
        raise NotImplementedError
    
    # Start inference
    start_time = time.time()
    
    if query == "single":
        # Initialize accumulators
        sum_images_per_query = 0.0
        sum_prompt_tokens_per_query = 0.0
        sum_completion_tokens_per_query = 0.0
        max_prompt_tokens_per_query = 0
        min_prompt_tokens_per_query = float("inf")
        max_completion_tokens_per_query = 0
        min_completion_tokens_per_query = float("inf")

        for i in range(DATASET_LEN):
            print(f"Processing dataset entry {i}/{DATASET_LEN}")
            if type in ["vllm", "huggingface"]:
                stats = inference_pipeline.run(
                    max_dataset_len=DATASET_LEN,
                    dataset_start_index=i
                )
            elif type == "api" and num_threads is not None:
                stats = inference_pipeline.run(
                    max_dataset_len=1,
                    dataset_start_index=i,
                    num_threads=num_threads
                )
            else:
                stats = inference_pipeline.run(
                    max_dataset_len=1,
                    dataset_start_index=i
                )

            # Accumulate values for averaging
            sum_images_per_query += stats["avg_images_per_query"]
            sum_prompt_tokens_per_query += stats["avg_prompt_tokens_per_query"]
            sum_completion_tokens_per_query += stats["avg_completion_tokens_per_query"]

            # Update min and max
            max_prompt_tokens_per_query = max(max_prompt_tokens_per_query, stats["max_prompt_tokens_per_query"])
            min_prompt_tokens_per_query = min(min_prompt_tokens_per_query, stats["min_prompt_tokens_per_query"])
            max_completion_tokens_per_query = max(max_completion_tokens_per_query, stats["max_completion_tokens_per_query"])
            min_completion_tokens_per_query = min(min_completion_tokens_per_query, stats["min_completion_tokens_per_query"])

        # Once the loop is done, compute the final averages:
        avg_images_per_query = sum_images_per_query / DATASET_LEN
        avg_prompt_tokens_per_query = sum_prompt_tokens_per_query / DATASET_LEN
        avg_completion_tokens_per_query = sum_completion_tokens_per_query / DATASET_LEN

    elif query == "batch":
        stats = inference_pipeline.run(max_dataset_len=DATASET_LEN)

    elif query == "all" and num_threads is not None:
        stats = inference_pipeline.run(max_dataset_len=DATASET_LEN, num_threads=num_threads)
    elif query == "all":
        stats = inference_pipeline.run(max_dataset_len=DATASET_LEN)
    else:
        raise NotImplementedError

    total_time = time.time() - start_time
    avg_time = total_time / DATASET_LEN

    if query == "single":
        results = {
            "total_time": total_time,
            "avg_time": avg_time,
            "avg_images_per_query": avg_images_per_query,
            "avg_prompt_tokens_per_query": avg_prompt_tokens_per_query,
            "avg_completion_tokens_per_query": avg_completion_tokens_per_query,
            "max_prompt_tokens_per_query": max_prompt_tokens_per_query,
            "min_prompt_tokens_per_query": min_prompt_tokens_per_query,
            "max_completion_tokens_per_query": max_completion_tokens_per_query,
            "min_completion_tokens_per_query": min_completion_tokens_per_query
        }
    else:
        results = {
            "total_time": total_time,
            "avg_time": avg_time,
            "avg_images_per_query": stats["avg_images_per_query"],
            "avg_prompt_tokens_per_query": stats["avg_prompt_tokens_per_query"],
            "avg_completion_tokens_per_query": stats["avg_completion_tokens_per_query"],
            "max_prompt_tokens_per_query": stats["max_prompt_tokens_per_query"],
            "min_prompt_tokens_per_query": stats["min_prompt_tokens_per_query"],
            "max_completion_tokens_per_query": stats["max_completion_tokens_per_query"],
            "min_completion_tokens_per_query": stats["min_completion_tokens_per_query"]
        }

    # Safe results to json and prints
    with open(file_path, "r") as f:
        data = json.load(f)

    if type == "api" and num_threads is not None:
        type = "api_with_threading" # Rename api dict key if threading
    
    if type not in data:
        data[type] = {}

    if query == "single" and images == "multiple":
        new_block_name = "single_query_multiple_images"
    elif query == "single" and images == "single":
        new_block_name = "single_query_single_image"
    elif query == "all" and images == "multiple":
        new_block_name = "multiple_queries_multiple_images"
    elif query == "all" and images == "single":
        new_block_name = "multiple_queries_single_image"
    elif query == "batch" and images == "single":
        new_block_name = f"batch_{batch_size}_query_single_image_sft"
    elif query == "batch" and images == "multiple":
        new_block_name = f"batch_{batch_size}_query_multiple_images_sft"
    data[type][new_block_name] = results

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Statistics saved in {file_path}")
    print(f"\nTotal time: {total_time}")
    print(f"Average time: {avg_time}")
    print(f"Average images per query: {stats['avg_images_per_query']}")
    print(f"Average prompt tokens per query: {stats['avg_prompt_tokens_per_query']}")
    print(f"Average completion tokens per query: {stats['avg_completion_tokens_per_query']}")
    print(f"Maximum completion tokens per query: {stats['max_completion_tokens_per_query']}")
    print(f"Minimum completion tokens per query: {stats['min_completion_tokens_per_query']}")
    print(f"Maximum prompt tokens per query: {stats['max_prompt_tokens_per_query']}")
    print(f"Minimum prompt tokens per query: {stats['min_prompt_tokens_per_query']}")



def modify_eval_dataset_single_image():
    path = "data/eval_dataset_single_image.json"
    with open(path, "r") as file:
        dataset = json.load(file)

    dataset = dataset[:500]
    
    with open(path, "w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)

def modify_eval_dataset_three_images():
    path = "data/eval_dataset_three_images.json"
    with open(path, "r") as file:
        dataset = json.load(file)

    for entry in dataset:
        entry["images"].append(entry["images"][0])
        entry["images"].append(entry["images"][0])
        entry["question"] = "<image><image>" + entry["question"]
    
    with open(path, "w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)


def _worker(bs: int) -> None:
    """
    Runs one complete inference pass in a fresh interpreter.
    Exits with code 0 on success, 1 on failure.
    """
    try:
        compute_and_safe_inference_statistics(
            file_path="stats/inference_stats/eval_dataset_inference_stats.json",
            type="huggingface",
            query="batch",
            images="single",
            batch_size=bs,
        )
    except Exception as exc:
        print(f"  ✗ Huggingface failed at batch_size={bs}: {exc}")
        traceback.print_exc()
        sys.exit(1)          # propagate the failure to the parent
    else:
        sys.exit(0)

if "__main__" == __name__:
    # For calculate inference time stats:
    BACKENDS   = ["chat"]
    QUERIES    = ["single", "all"]
    IMAGE_MODES = ["multiple", "single"]

    # for backend in BACKENDS:
    #     for query in QUERIES:
    #         for images in IMAGE_MODES:
    #             print(f"→ Running {backend=} {query=} {images=}")
    #             kwargs = {
    #                 "file_path": "stats/inference_stats/eval_dataset_inference_stats.json",
    #                 "type":      backend,
    #                 "query":     query,
    #                 "images":    images,
    #             }
    #             if backend == "api":
    #                 # only API needs num_threads
    #                 kwargs["num_threads"] = 4

    #             compute_and_safe_inference_statistics(**kwargs)


    for batch_size in [1, 2, 4, 8, 16, 32]:
        try:
            compute_and_safe_inference_statistics(file_path="stats/inference_stats/eval_dataset_inference_stats.json", type="vllm", query="batch", images="single", batch_size=batch_size)
        except Exception as e:
            # log the failure and continue
            print(f"  ✗ vLLM failed at batch_size={batch_size}: {e}")
        else:
            print(f"  ✓ vLLM succeeded at batch_size={batch_size}")

    mp.set_start_method("spawn", force=True)

    for batch_size in [1, 2, 4, 8, 16, 32]:
        proc = mp.Process(target=_worker, args=(batch_size,))
        proc.start()
        proc.join()

        if proc.exitcode == 0:
            print(f"  ✓ Huggingface succeeded at batch_size={batch_size}")
        else:
            print(f"  ✗ Huggingface failed at batch_size={batch_size}")

