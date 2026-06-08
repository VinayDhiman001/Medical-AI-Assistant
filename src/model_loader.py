import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class ModelLoader:
    """
    Responsible for loading and preparing model for fine-tuning.
    """
    
    def __init__(self, config: dict):
        self.config = config
    
    def load(self, hf_token: str):
        """Loads model with 4-bit quantization"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['name'], token=hf_token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['name'],
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token
        )
        return self
    
    def prepare_for_training(self):
        """Adds LoRA adapters for QLoRA fine-tuning"""
        self.model = prepare_model_for_kbit_training(self.model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        return self
