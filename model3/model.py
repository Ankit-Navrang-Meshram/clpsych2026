#model.py

from transformers import Qwen2Model, Qwen2PreTrainedModel, AutoTokenizer , BitsAndBytesConfig , AutoModelForCausalLM
import torch
import torch.nn as nn




class QwenPredictor:
    def __init__(self, model_name: str, device: str  = "cuda", load_in_8bit: bool = False, max_new_tokens: int  = 256) -> None:
        print(f"[QwenPredictor] Loading {model_name} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            quantization_config=BitsAndBytesConfig(load_in_8bit = True),
            trust_remote_code=True,
        )
        self.model.eval()
        self.device         = device
        self.max_new_tokens = max_new_tokens
        print("[QwenPredictor] Ready.")