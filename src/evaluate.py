from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from config import MODEL_OUTPUT_DIR, PROCESSED_DATASET_FILE
import math

def evaluate_model(
    model_dir: str = MODEL_OUTPUT_DIR,
    dataset_path: str = PROCESSED_DATASET_FILE,
    max_length: int = 1024,
    per_device_eval_batch_size: int = 2,
):
    """
    Evaluate the fine-tuned model on a test dataset.

    Args:
        model_dir (str): Path to the fine-tuned model directory.
        dataset_path (str): Path to the processed dataset.
        max_length (int): Maximum sequence length for evaluation.
        per_device_eval_batch_size (int): Batch size for evaluation.
    """
    # Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load the dataset
    dataset = load_from_disk(dataset_path)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Define evaluation arguments
    eval_args = TrainingArguments(
        output_dir="./eval_results",
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