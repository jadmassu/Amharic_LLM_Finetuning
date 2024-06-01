import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer

class AmharicLlama2FineTuner:
    def __init__(self, base_model, dataset_name, fine_tuned_model, quant_config, peft_params, training_params):
        self.base_model = base_model
        self.dataset_name = dataset_name
        self.fine_tuned_model = fine_tuned_model
        self.quant_config = quant_config
        self.peft_params = peft_params
        self.training_params = training_params
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
    
    def load_dataset(self):
        self.dataset =  Dataset.load_from_disk(self.dataset_name,split="train")
    
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=self.quant_config,
            device_map={"": 0}
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
    
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    
    def initialize_trainer(self):
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            peft_config=self.peft_params,
            dataset_text_field="text",
            max_seq_length=None,
            tokenizer=self.tokenizer,
            args=self.training_params,
            packing=False,
        )
    
    def train(self):
        self.trainer.train()
        self.trainer.save_model(self.fine_tuned_model)
    
    def fine_tune(self):
        self.load_dataset()
        self.load_model()
        self.load_tokenizer()
        self.initialize_trainer()
        self.train()
