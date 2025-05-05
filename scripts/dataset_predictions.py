from utils.inference_api import InferenceAPI
from utils.inference_chat import InferenceChat
from utils.inference_vllm import InferenceVLLM

def predict_inference_api(model_config_path, dataset_path, json_output_path, csv_output_path):
    inference_pipeline = InferenceAPI(
        model_config_path=model_config_path,
        dataset_path=dataset_path,
        json_output_path=json_output_path,
        csv_output_path=csv_output_path,
    )
    inference_pipeline.run(num_threads = 4)
    inference_pipeline.save_json()
    inference_pipeline.save_csv()


def predict_inference_chat(model_config_path, dataset_path, json_output_path, csv_output_path):
    inference_pipeline = InferenceChat(
        model_config_path=model_config_path,
        dataset_path=dataset_path,
        json_output_path=json_output_path,
        csv_output_path=csv_output_path,
    )
    inference_pipeline.run()
    inference_pipeline.save_json()
    inference_pipeline.save_csv()


def predict_inference_vllm(
    model_name_or_path="Qwen/Qwen2-VL-7B-Instruct",
    dataset="lingoQA_scenery_100",
    template="qwen2_vl",
    csv_output_path="./data/predictions/lingoQA_scenery_100_predictions.csv",
    json_output_path="./data/predictions/lingoQA_scenery_100_predictions.json",
    adapter_name_or_path=None,  # or "" if you prefer
):

    inference_pipeline = InferenceVLLM(
        model_name_or_path=model_name_or_path,
        dataset=dataset,
        template=template,
        csv_output_path=csv_output_path,
        json_output_path=json_output_path,
        adapter_name_or_path=adapter_name_or_path,
    )
    inference_pipeline.run()
    inference_pipeline.save_json()
    inference_pipeline.save_csv()


if "__main__" == __name__:

    predict_inference_chat(
            model_config_path="inference_configs/fixed_inference_image_resolution_589824/qwen2vl_lora_sft_lingoQA_scenery_10000.yaml",
            dataset_path="data/lingoQA_evaluation.json",
            json_output_path="data/predictions/chat_backend/fixed_inference_image_resolution_589824/qwen2vl_lora_sft_scenery_10000_lingoQA_evaluation_dataset.json",
            csv_output_path="data/predictions/chat_backend/fixed_inference_image_resolution_589824/qwen2vl_lora_sft_scenery_10000_lingoQA_evaluation_dataset.csv",
    )

    predict_inference_chat(
            model_config_path="inference_configs/adapted_inference_image_resolution/qwen2vl_base.yaml",
            dataset_path="data/lingoQA_evaluation.json",
            json_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_base_lingoQA_evaluation_dataset.json",
            csv_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_base_lingoQA_evaluation_dataset.csv",
    )

    predict_inference_chat(
            model_config_path="inference_configs/adapted_inference_image_resolution/qwen2vl_lora_sft_lingoQA_action_1000.yaml",
            dataset_path="data/lingoQA_evaluation.json",
            json_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_action_1000_lingoQA_evaluation_dataset.json",
            csv_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_action_1000_lingoQA_evaluation_dataset.csv",
    )

    predict_inference_chat(
            model_config_path="inference_configs/adapted_inference_image_resolution/qwen2vl_lora_sft_lingoQA_action_10000.yaml",
            dataset_path="data/lingoQA_evaluation.json",
            json_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_action_10000_lingoQA_evaluation_dataset.json",
            csv_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_action_10000_lingoQA_evaluation_dataset.csv",
    )

    predict_inference_chat(
            model_config_path="inference_configs/adapted_inference_image_resolution/qwen2vl_lora_sft_lingoQA_action_10000_high_res.yaml",
            dataset_path="data/lingoQA_evaluation.json",
            json_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_action_10000_high_res_lingoQA_evaluation_dataset.json",
            csv_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_action_10000_high_res_lingoQA_evaluation_dataset.csv",
    )

    predict_inference_chat(
            model_config_path="inference_configs/adapted_inference_image_resolution/qwen2vl_lora_sft_lingoQA_action_25000.yaml",
            dataset_path="data/lingoQA_evaluation.json",
            json_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_action_25000_lingoQA_evaluation_dataset.json",
            csv_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_action_25000_lingoQA_evaluation_dataset.csv",
    )

    predict_inference_chat(
            model_config_path="inference_configs/adapted_inference_image_resolution/qwen2vl_lora_sft_lingoQA_action_and_scenery_20000_high_res.yaml",
            dataset_path="data/lingoQA_evaluation.json",
            json_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_action_and_scenery_20000_high_res_lingoQA_evaluation_dataset.json",
            csv_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_action_and_scenery_20000_high_res_lingoQA_evaluation_dataset.csv",
    )

    predict_inference_chat(
            model_config_path="inference_configs/adapted_inference_image_resolution/qwen2vl_lora_sft_lingoQA_scenery_1000.yaml",
            dataset_path="data/lingoQA_evaluation.json",
            json_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_scenery_1000_lingoQA_evaluation_dataset.json",
            csv_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_scenery_1000_lingoQA_evaluation_dataset.csv",
    )

    predict_inference_chat(
            model_config_path="inference_configs/adapted_inference_image_resolution/qwen2vl_lora_sft_lingoQA_scenery_10000.yaml",
            dataset_path="data/lingoQA_evaluation.json",
            json_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_scenery_10000_lingoQA_evaluation_dataset.json",
            csv_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_scenery_10000_lingoQA_evaluation_dataset.csv",
    )

    predict_inference_chat(
            model_config_path="inference_configs/adapted_inference_image_resolution/qwen2vl_lora_sft_lingoQA_scenery_10000_high_res.yaml",
            dataset_path="data/lingoQA_evaluation.json",
            json_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_scenery_10000_high_res_lingoQA_evaluation_dataset.json",
            csv_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_scenery_10000_high_res_lingoQA_evaluation_dataset.csv",
    )

    predict_inference_chat(
            model_config_path="inference_configs/adapted_inference_image_resolution/qwen2vl_lora_sft_lingoQA_scenery_25000.yaml",
            dataset_path="data/lingoQA_evaluation.json",
            json_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_scenery_25000_lingoQA_evaluation_dataset.json",
            csv_output_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_scenery_25000_lingoQA_evaluation_dataset.csv",
    )