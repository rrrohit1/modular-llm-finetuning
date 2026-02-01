import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
# Change 1: Import SFTConfig instead of TrainingArguments
from trl import SFTTrainer, SFTConfig 
from src.config import Config
from src.data_prep import prepare_data

def train():
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    dataset = prepare_data(tokenizer)

    # 1. Keep SFTConfig for standard training hyperparams
    training_args = SFTConfig(
        output_dir=Config.OUTPUT_DIR,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        num_train_epochs=Config.NUM_EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        # Remove max_seq_length and dataset_text_field from here
    )

    # 2. Pass the dataset-specific arguments HERE instead
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(f"{Config.OUTPUT_DIR}/final_adapter")

if __name__ == "__main__":
    train()