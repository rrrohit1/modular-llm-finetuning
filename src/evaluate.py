from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from config import MODEL_OUTPUT_DIR, PROCESSED_DATASET_FILE, DATASET_PATHS
import math

def evaluate_model(
    model_dir: str = MODEL_OUTPUT_DIR,
    dataset_name: str = "dataset1",
    max_length: int = 1024,
    per_device_eval_batch_size: int = 2,
):
    """
    Evaluate the fine-tuned model on a specified dataset.

    Args:
        model_dir (str): Path to the fine-tuned model directory.
        dataset_name (str): Name of the dataset to use for evaluation.
        max_length (int): Maximum sequence length for evaluation.
        per_device_eval_batch_size (int): Batch size for evaluation.
    """
    # Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load the dataset
    dataset_path = DATASET_PATHS.get(dataset_name)
    if not dataset_path:
        raise ValueError(f"Dataset {dataset_name} not found in configuration.")

    dataset = load_dataset("csv", data_files=dataset_path)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "response"])

    # Define evaluation arguments
    eval_args = TrainingArguments(
        output_dir=f"./eval_results/{dataset_name}",
        per_device_eval_batch_size=per_device_eval_batch_size,
        logging_dir="./logs",
        report_to="none",
    )

    # Initialize the Trainer for evaluation
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Evaluate the model
    eval_results = trainer.evaluate()

    # Calculate perplexity
    perplexity = math.exp(eval_results["eval_loss"])
    print(f"Perplexity: {perplexity}")

    return eval_results


if __name__ == "__main__":
    results = evaluate_model()
    print("Evaluation Results:", results)