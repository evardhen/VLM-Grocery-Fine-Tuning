from utils.inference_api import InferenceAPI
from utils.inference_chat import InferenceChat
from utils.inference_vllm import InferenceVLLM

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
    dataset="lingoQA_scenery_100",
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


if "__main__" == __name__:
    predict_inference_chat(
            model_config_path="configs/inference_configs/adapted_inference_image_resolution/qwen2vl_lora_sft_grocery_262144_res.yaml",
            dataset_path="data/eval.json",
            json_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_grocery_262144_res.json",
    )