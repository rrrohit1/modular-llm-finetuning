import torch

class Config:
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    DATASETS = ["dataset_1.jsonl", "dataset_2.jsonl", "dataset_3.jsonl"] # Replace with your paths
    OUTPUT_DIR = "./qwen-lora-output"
    
    # LoRA Hyperparameters
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Training Hyperparameters
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    MAX_SEQ_LENGTH = 1024