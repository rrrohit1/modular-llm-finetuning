import os
from datasets import load_dataset, concatenate_datasets
from src.config import Config
from src.prompts import format_github_issue

def prepare_data(tokenizer, number_of_samples=1000):
    """
    Loads Embold dataset and other datasets, merges them, and applies chat templates.
    """
    # 1. Load the specific Embold datasets
    # Path: data/raw/embold_train.json
    base_path = "data/raw"
    
    # Loading Train and Train_extra
    dataset = load_dataset("json", data_files=os.path.join(base_path, "embold_train.json"), split="train", )
    

    def apply_template(example):
        # Use the prompt formatter from prompts.py
        messages = format_github_issue(example)
        
        # Apply the chat template for Qwen
        example["text"] = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False # False because we include the assistant label for training
        )
        return example

    # Shuffle and map
    dataset = dataset.shuffle(seed=2026).select(range(min(number_of_samples, len(dataset)))).map(
        apply_template,
        remove_columns=dataset.column_names, # Clean up raw columns to save memory
        desc="Applying Chat Template"
    )

    return dataset