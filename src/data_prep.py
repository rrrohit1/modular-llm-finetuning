import os
from datasets import load_dataset, concatenate_datasets
from src.config import Config
from src.prompts import format_github_issue

def prepare_data(tokenizer):
    """
    Loads Embold dataset and other datasets, merges them, and applies chat templates.
    """
    # 1. Load the specific Embold datasets
    # Path: data/raw/embold_train.json
    base_path = "data/raw"
    
    # Loading Train and Train_extra
    ds_main = load_dataset("json", data_files=os.path.join(base_path, "embold_train.json"), split="train")
    
    # Check if extra training data exists and append it
    extra_path = os.path.join(base_path, "embold_train_extra.json")
    if os.path.exists(extra_path):
        ds_extra = load_dataset("json", data_files=extra_path, split="train")
        dataset = concatenate_datasets([ds_main, ds_extra])
    else:
        dataset = ds_main

    # 2. Add other datasets from Config if they exist
    # (Assuming they follow the same GitHub structure or you have logic for them)
    # ... logic for other datasets can be added here ...

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
    dataset = dataset.shuffle(seed=42).map(
        apply_template,
        remove_columns=dataset.column_names, # Clean up raw columns to save memory
        desc="Applying Chat Template"
    )

    return dataset