import json
import math
from transformers import (
    Seq2SeqTrainingArguments,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    GenerationConfig,
)
from peft import PeftModel  # pip install peft
import torch

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.packages import is_pillow_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject


class InferenceHuggingface:
    """
    A class to run inference using vLLM engine, plus helper methods for saving
    data as JSON.
    """
    def __init__(        
        self,
        model_name_or_path: str,
        dataset: str,
        template: str,
        json_output_path: str,
        adapter_name_or_path: str = None,
        dataset_dir: str = "data",
        cutoff_len: int = 8192,
        max_samples: int = None,
        vllm_config: str = "{}",
        temperature: float = 0.95, # Randomness of output sampling
        top_p: float = 0.7, # Sorting tokens by their probability in a descending order and then adding them up, until top_p value is reached; then softmax is calculated with reduced vocab size
        top_k: int = 50, # Same as top_p, but with highest k prob values
        max_new_tokens: int = 1024,
        repetition_penalty: float = 1.0, # Lower 1 penalizes same output tokens, larger 1 enhances it
        image_resolution: int = 589824): 
        
        self.predictions = []
        self.json_output_path = json_output_path

        self.max_dataset_len = max_samples
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.vllm_config = vllm_config
        self.cutoff_len = cutoff_len
        self.dataset_dir = dataset_dir
        self.adapter_name_or_path = adapter_name_or_path
        self.template = template
        self.dataset = dataset
        self.model_name_or_path = model_name_or_path
        self.image_resolution = image_resolution
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")


    def run(self, max_dataset_len = None, dataset_start_index = None):
        """
        Performs batch generation using the vLLM engine, which supports tensor parallelism.
        """
        self.max_dataset_len = max_dataset_len or self.max_dataset_len
        total_images = 0
        prompt_token_counts = []
        completion_token_counts = []
        self.predictions = []  

        self.model_args, self.data_args, _, self.generating_args = get_infer_args(
            dict(
                model_name_or_path=self.model_name_or_path,
                image_max_pixels=self.image_resolution,
                adapter_name_or_path=self.adapter_name_or_path,
                dataset=self.dataset,
                dataset_dir=self.dataset_dir,
                template=self.template,
                cutoff_len=self.cutoff_len,
                max_samples=self.max_dataset_len,
                preprocessing_num_workers=40,
                vllm_config=self.vllm_config,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=self.repetition_penalty,
            )
        )

        training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
        tokenizer_module = load_tokenizer(self.model_args)
        tokenizer = tokenizer_module["tokenizer"]
        template_obj = get_template_and_fix_tokenizer(tokenizer, self.data_args)
        # Fix for qwen2_vl <|image_pad|> token expanding
        template_obj.mm_plugin.expand_mm_tokens = False
        dataset_module = get_dataset(template_obj, self.model_args, self.data_args, training_args, "ppo", **tokenizer_module)

        inputs = []
        if dataset_start_index is not None:
            dataset_module["train_dataset"] = dataset_module["train_dataset"].select(range(dataset_start_index, dataset_start_index + 1))
        print(f"Processing dataset of size {dataset_module['train_dataset'].num_rows}")

        for sample in dataset_module["train_dataset"]:
            if sample["images"]:
                total_images += len(sample["images"])
                multi_modal_data = {"image": []}
                for image in sample["images"]:
                    if not isinstance(image, (str, ImageObject)):
                        raise ValueError(f"Expected image input as path or PIL.Image, but got {type(image)}.")
                    if isinstance(image, str):
                        image = Image.open(image).convert("RGB")
                        # Manually resize image
                        image = self._preprocess_image(image, self.model_args.image_max_pixels)
                    multi_modal_data["image"].append(image)
            else:
                multi_modal_data = None

            inputs.append({"prompt_token_ids": sample["input_ids"], "multi_modal_data": multi_modal_data})

        # --- HuggingFace equivalent of your sampling_params ---
        gen_config = GenerationConfig(
            temperature            = self.generating_args.temperature,
            top_p                  = self.generating_args.top_p or 1.0,
            top_k                  = self.generating_args.top_k,
            repetition_penalty     = self.generating_args.repetition_penalty or 1.0,
            max_new_tokens         = self.generating_args.max_new_tokens,
            eos_token_id           = tokenizer.eos_token_id,
            pad_token_id           = tokenizer.pad_token_id,
        )

        # --- load base model (Seq2Seq or Causal) ---
        try:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
        except Exception:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )

        # --- wrap in LoRA if requested ---
        if self.adapter_name_or_path:
            model = PeftModel.from_pretrained(base_model, self.adapter_name_or_path)
        else:
            model = base_model
        model.to(self.device)
        model.eval()    

        # Run inference
        with torch.no_grad():
            results = model.generate(
                **inputs,
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=False,
            )
        sequences = results.sequences  
        preds = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        print(f"\nProcessed {len(preds)} prompts with HF generate.\n")
 
        dataset_path = "./" + self.data_args.dataset_dir + "/" + self.data_args.dataset[0] + ".json"
         
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        if dataset_start_index is not None:
            dataset = dataset[dataset_start_index:dataset_start_index+1]
        elif self.max_dataset_len < len(dataset):
            dataset = dataset[:self.max_dataset_len]
        len_dataset = len(dataset)

        for entry, result in zip(dataset, preds):
            self.predictions.append({
                "question_id": entry["question_id"],
                "question": entry["question"],
                "answer": result,
            })

        # Compute statistics
        for i, seq in enumerate(sequences):
            # 1) prompt length
            #    count all tokens != pad_token_id in the input
            prompt_len = (inputs["input_ids"][i] != tokenizer.pad_token_id).sum().item()
            prompt_token_counts.append(prompt_len)

            # 2) completion length
            #    for encoder-decoder (BART/T5/…): seq is just generated tokens
            #    for decoder-only (GPT style): seq includes the prompt, so subtract
            if hasattr(model.config, "is_encoder_decoder") and model.config.is_encoder_decoder:
                completion_len = seq.size(-1)
            else:
                completion_len = seq.size(-1) - prompt_len

            # just in case somebody passes a shorter output:
            completion_len = max(completion_len, 0)
            completion_token_counts.append(completion_len)

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

    def save_json(self):
        try:
            with open(self.json_output_path, "w", encoding="utf-8") as f:
                json.dump(self.predictions, f, indent=4)
            print(f"Predictions saved to {self.json_output_path}")
        except Exception as e:
            print(f"Failed to save JSON. Error: {e}")

    def _preprocess_image(self, image: "ImageObject", image_resolution) -> "ImageObject":
        r"""
        Pre-processes a single image.
        """
        if (image.width * image.height) > image_resolution:
            resize_factor = math.sqrt(image_resolution / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image
    

if __name__ == "__main__":
    def predict_inference_huggingface(
        model_name_or_path="Qwen/Qwen2-VL-7B-Instruct",
        dataset="data/eval.json",
        template="qwen2_vl",
        json_output_path="data/predictions/chat_backend/fixed_inference_image_resolution_589824/test.json",
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

    predict_inference_huggingface()