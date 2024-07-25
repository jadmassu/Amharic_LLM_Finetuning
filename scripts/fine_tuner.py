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
    def __init__(self, base_model_path, dataset_name, fine_tuned_model_name):
        self.base_model = base_model_path
        self.dataset_name = dataset_name
        self.fine_tuned_model = fine_tuned_model_name
      
     
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        # LoRA attention dimension
        self.lora_r = 64
        # Alpha parameter for LoRA scaling
        self.lora_alpha = 16
        # Dropout probability for LoRA layers
        self.lora_dropout = 0.1

    def quant_config(self):
        bnb_4bit_compute_dtype = "float16"
        
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
)

    def load_dataset(self):
        self.dataset =  Dataset.load_from_disk(self.dataset_name,split="train")
    
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=self.quant_config(),
            device_map={"": 0}
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
       
       # Load LoRA configuration
    def lora_config(self):
       return  LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        # embedding_size = self.base_model.get_input_embeddings().weight.shape[0]
    def training_arguments(self):
        return TrainingArguments(
            output_dir= "./results",
            num_train_epochs= 1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_steps= 0,
            logging_steps=25,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16= False,
            bf16= False,
            max_grad_norm=0.3,
            max_steps= -1,
            warmup_ratio= 0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="tensorboard"
        )
    
    def initialize_trainer(self):
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=None,
            tokenizer=self.tokenizer,
            peft_config= self.lora_config(),
            args=self.training_arguments(),
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
