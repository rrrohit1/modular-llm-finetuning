from datasets import load_dataset, concatenate_datasets
from src.config import Config

def prepare_data(tokenizer):
    all_datasets = []
    for data_path in Config.DATASETS:
        # Assuming JSONL format, adjust if using CSV/HuggingFace Hub
        ds = load_dataset("json", data_files=data_path, split="train")
        all_datasets.append(ds)
    
    merged_dataset = concatenate_datasets(all_datasets)
    
    def apply_template(example):
        from src.prompts import format_instruction
        messages = format_instruction(example)
        # Apply the chat template without tokenizing yet
        example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
        return example

    return merged_dataset.map(apply_template)