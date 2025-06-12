import json
import math
from pathlib import Path
from PIL import Image
from transformers import (
    GenerationConfig,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from peft import PeftModel  # pip install peft
import torch
from accelerate import find_executable_batch_size, load_checkpoint_and_dispatch, init_empty_weights
import contextlib
from qwen_vl_utils import process_vision_info


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
        cutoff_len: int = 1024,
        max_samples: int = None,
        vllm_config: str = "{}",
        temperature: float = 0.95, # Randomness of output sampling
        top_p: float = 0.7, # Sorting tokens by their probability in a descending order and then adding them up, until top_p value is reached; then softmax is calculated with reduced vocab size
        top_k: int = 50, # Same as top_p, but with highest k prob values
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.0, # Lower 1 penalizes same output tokens, larger 1 enhances it
        image_resolution: int = 262144,
        batch_size = 8): 
        
        self.predictions = []
        self.json_output_path = json_output_path

        self.max_dataset_len = max_samples
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
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
    
    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        if (img.width * img.height) > self.image_resolution:
            f = math.sqrt(self.image_resolution / (img.width * img.height))
            img = img.resize((int(img.width * f), int(img.height * f)),
                             resample=Image.NEAREST)
        return img
    
    def _load_samples(self, start_idx: int | None) -> list[dict]:
        dataset_path = Path(self.dataset_dir) / f"{self.dataset}.json"
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # slice the dataset as requested
        if start_idx is not None:
            data = data[start_idx : start_idx + 1]
        elif self.max_dataset_len and len(data) > self.max_dataset_len:
            data = data[: self.max_dataset_len]

        samples = []
        for row in data:
            prompt = row["question"]            # <-- adjust if your key differs
            images = [self._load_image(p) for p in row.get("images", [])]
            samples.append(
                {"prompt": prompt,
                 "images": images,
                 "question_id": row["question_id"],
                 "system": row["system"]}
            )
        return samples


    def run(self, max_dataset_len = None, dataset_start_index = None):
        """
        Performs batch generation using the vLLM engine, which supports tensor parallelism.
        """
        self.max_dataset_len = max_dataset_len or self.max_dataset_len
        self.predictions = []  
        prompt_token_counts, completion_token_counts = [], []


        # --- HuggingFace equivalent of your sampling_params ---
        gen_config = GenerationConfig(
            temperature            = self.temperature,
            top_p                  = self.top_p or 1.0,
            top_k                  = self.top_k,
            repetition_penalty     = self.repetition_penalty or 1.0,
            max_new_tokens         = self.max_new_tokens,
        )

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16,
        )

        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        model = (PeftModel.from_pretrained(base_model,
                                           self.adapter_name_or_path,
                                           is_trainable=False,
                                           torch_dtype=torch.float16)
                 if self.adapter_name_or_path else base_model)
        model.eval()

        samples = self._load_samples(dataset_start_index)
        if not samples:
            raise RuntimeError("No samples found ― check dataset path/keys")
        SYSTEM_PROMPT = samples[0]["system"]

        messages = []
        chunk_start = 0

        for idx, sample in enumerate(samples):
            content = [
                {"type": "image", "image": img}          # one dict *per* image
                for img in sample["images"]
            ]
            content.append({"type": "text", "text": sample["prompt"]})
            message = [{"role": "user", "content": content}]
            messages.append(message)

            flush = (len(messages) == self.batch_size) or (idx == len(samples) - 1)
            if not flush:
                continue

            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, system_message=SYSTEM_PROMPT, add_generation_prompt=True)
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs,
                                        generation_config=gen_config,
                                        return_dict_in_generate=False,
                                        do_sample=True)
                
            # preds = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            preds = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
                            # ---- stats ------------------------------------------
            plens = (inputs.input_ids
                        != self.processor.tokenizer.pad_token_id).sum(dim=1)
            clen  = generated_ids.shape[1] - plens
            prompt_token_counts.extend(plens.tolist())
            completion_token_counts.extend(clen.tolist())
            for s, pred in zip(samples[chunk_start:idx + 1], preds):
                self.predictions.append(
                    {"question_id": s["question_id"], "question": s["prompt"], "answer": pred}
                )
            torch.cuda.empty_cache()
            messages.clear()
            chunk_start = idx + 1          # next slice
                    


        # ---- 5.  Aggregate stats -------------------------------------
        self.stats = {
            "avg_images_per_query": len(samples[0]["images"]),
            "avg_prompt_tokens_per_query":     sum(prompt_token_counts)     / len(prompt_token_counts),
            "avg_completion_tokens_per_query": sum(completion_token_counts) / len(completion_token_counts),
            "max_prompt_tokens_per_query":     max(prompt_token_counts),
            "min_prompt_tokens_per_query":     min(prompt_token_counts),
            "max_completion_tokens_per_query": max(completion_token_counts),
            "min_completion_tokens_per_query": min(completion_token_counts),
        }
        return self.stats

    def save_json(self):
        try:
            with open(self.json_output_path, "w", encoding="utf-8") as f:
                json.dump(self.predictions, f, indent=4)
            print(f"Predictions saved to {self.json_output_path}")
        except Exception as e:
            print(f"Failed to save JSON. Error: {e}")
    