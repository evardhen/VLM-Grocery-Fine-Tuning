from utils.inference_api import InferenceAPI
from utils.inference_chat import InferenceChat
from utils.inference_vllm import InferenceVLLM
from utils.inference_huggingface import InferenceHuggingface
import glob
import os

def predict_inference_api(model_config_path, dataset_path, json_output_path):
    inference_pipeline = InferenceAPI(
        model_config_path=model_config_path,
        dataset_path=dataset_path,
        json_output_path=json_output_path,
    )
    inference_pipeline.run(num_threads = 4)
    inference_pipeline.save_json()


def predict_inference_chat(model_config_path, dataset_path, json_output_path):
    inference_pipeline = InferenceChat(
        model_config_path=model_config_path,
        dataset_path=dataset_path,
        json_output_path=json_output_path,
    )
    inference_pipeline.run()
    inference_pipeline.save_json()


def predict_inference_vllm(
    model_name_or_path="Qwen/Qwen2-VL-7B-Instruct",
    dataset="data/eval.json",
    template="qwen2_vl",
    json_output_path="./data/predictions/lingoQA_scenery_100_predictions.json",
    adapter_name_or_path=None,  # or "" if you prefer
):

    inference_pipeline = InferenceVLLM(
        model_name_or_path=model_name_or_path,
        dataset=dataset,
        template=template,
        json_output_path=json_output_path,
        adapter_name_or_path=adapter_name_or_path,
    )
    inference_pipeline.run()
    inference_pipeline.save_json()

def predict_inference_huggingface(
    model_name_or_path="Qwen/Qwen2-VL-7B-Instruct",
    dataset="data/eval.json",
    template="qwen2_vl",
    json_output_path="data/predictions/chat_backend/fixed_inference_image_resolution_589824",
    adapter_name_or_path=None,  # or "" if you prefer
):

    inference_pipeline = InferenceHuggingface(
        model_name_or_path=model_name_or_path,
        dataset=dataset,
        template=template,
        json_output_path=json_output_path,
        adapter_name_or_path=adapter_name_or_path,
    )
    inference_pipeline.run()
    inference_pipeline.save_json()


if "__main__" == __name__:
    config_dir = "configs/inference_configs/fixed_inference_image_resolution_589824"
    output_dir = "data/predictions/chat_backend/fixed_inference_image_resolution_589824"
    dataset_path = "data/eval.json"
    config_files = glob.glob(os.path.join(config_dir, "*.yaml"))

    # Loop over each config file
    for config_path in config_files:
        base_filename = os.path.basename(config_path)            

        # Extract the base filename (without extension) to name the output JSON
        base_name = os.path.splitext(base_filename)[0]
        json_output_path = os.path.join(output_dir, f"{base_name}.json")

        predict_inference_chat(
            model_config_path=config_path,
            dataset_path=dataset_path,
            json_output_path=json_output_path,
        )