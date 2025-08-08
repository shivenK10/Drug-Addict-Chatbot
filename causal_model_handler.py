import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelHandler:
    def __init__(self, model_name: str, quantize: bool = False):
        """
        Initialization of class arguments.

        1. model_name -> str -> Hugging face repo id.
        2. quantize -> bool -> Whether to load the model in float16.
        """
        self.model_name = model_name
        self.quantize = quantize
    
    def load_model(self):
        """
        Loads the tokenizer and causal LM model.

        Returns:
            model: The loaded AutoModelForCausalLM on CPU.
            tokenizer: The corresponding AutoTokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        
        if self.quantize:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map="cpu",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                device_map="cpu",
            )
        
        return model, tokenizer
