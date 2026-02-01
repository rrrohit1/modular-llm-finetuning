import argparse
import os
import sys
from pathlib import Path

# Add project root to path for internal imports
project_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(project_root))

from src.config import Config
from src.fine_tune import train
# Note: If you still need to download the GitHub dataset automatically, 
# you would import that utility here.

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run LoRA fine-tuning pipeline for GitHub Issue Classification'
    )
    
    # Model & Data Overrides
    parser.add_argument('--model_name', type=str, default=Config.MODEL_NAME,
                      help=f'Model to fine-tune (default: {Config.MODEL_NAME})')
    parser.add_argument('--dataset_path', type=str, default=Config.bug_dataset_path,
                      help=f'Path to JSON dataset (default: {Config.bug_dataset_path})')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                      help='Training batch size per device')
    parser.add_argument('--learning_rate', type=float, default=Config.LEARNING_RATE,
                      help='Learning rate')
    
    # Pipeline control
    parser.add_argument('--skip_prepare', action='store_true',
                      help='Skip data preparation (not recommended if data is raw)')
    
    return parser.parse_args()

def sync_config(args):
    """Update the Config class attributes with CLI arguments at runtime."""
    Config.MODEL_NAME = args.model_name
    Config.bug_dataset_path = args.dataset_path
    Config.NUM_EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.learning_rate

def validate_environment():
    """Ensure data exists before starting."""
    data_path = Path(Config.bug_dataset_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please ensure 'embold_train.json' is in data/raw/"
        )
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

def main():
    args = parse_args()
    
    # 1. Sync CLI args with the Config class
    sync_config(args)
    
    # 2. Validate paths
    print(f"--- Pipeline Starting ---")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Data:  {Config.bug_dataset_path}")
    validate_environment()

    # 3. Fine-tune the model
    # We call the 'train' function from src/fine_tune.py
    # Since 'train()' internally uses 'Config' and 'prepare_data()', 
    # it will pick up our overrides.
    print("\nStep: Starting LoRA Fine-tuning...")
    try:
        train()
        print(f"\nTraining Complete! Model saved to {Config.OUTPUT_DIR}/final_adapter")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()