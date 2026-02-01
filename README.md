# Fine-tuning Language Models on Medical Transcriptions

This project fine-tunes large language models (LLaMA and Gemma) on medical transcriptions to improve their capabilities in medical text generation and understanding. The project supports both Google's Gemma 2B and Meta's LLaMA models.

## Project Structure

```bash
├── data/
│   ├── raw/              # Raw medical transcriptions data
│   │   └── embold_train.json
│   └── processed/        # Processed dataset for training
├── models/              # Directory for saving trained models
├── notebooks/          
│   └── finetune-gemma1b.ipynb    # Fine-tuning notebook
├── src/
│   ├── config.py        # Configuration and training parameters
│   ├── data_prep.py     # Data preparation and tokenization
│   ├── download_dataset.py  # Dataset download script
│   ├── fine_tune.py     # Fine-tuning script using SFTTrainer and LoRA
│   ├── evaluate.py      # Model evaluation script
│   ├── inference.py     # Inference script
│   ├── prompts.py       # Prompt templates
│   └── main.py          # Main pipeline orchestrator
└── environment.yaml     # Conda environment file
```

## Setup

1. Create and activate the conda environment:

```bash
conda env create -f environment.yaml
conda activate gemma-env
```

1. Install required packages:

```bash
pip install transformers datasets pandas numpy matplotlib seaborn python-dotenv kaggle jupyter notebook bitsandbytes peft trl
```

1. Set up Kaggle credentials:
   - Get your Kaggle API token from your Kaggle account settings
   - Create a `.env` file in the project root with:

```bash
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

## Usage

### Running the Fine-Tuning Pipeline

The complete fine-tuning pipeline can be run using the main script:

```bash
python src/main.py
```

Optional arguments:

- `--model_name`: Model to fine-tune (default: from `Config.MODEL_NAME`)
- `--dataset_path`: Path to the dataset file (optional)
- `--skip_download`: Skip downloading the dataset

### Example Commands

```bash
# Run with default configuration
python src/main.py

# Skip dataset download
python src/main.py --skip_download

# Specify a different model
python src/main.py --model_name "meta-llama/Llama-2-7b-hf"
```

### Running Individual Components

1. **Prepare Data**:
   ```bash
   python -c "from src.data_prep import prepare_data; from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b'); prepare_data(tokenizer)"
   ```

2. **Fine-Tune the Model**:
   ```bash
   python src/fine_tune.py
   ```

3. **Evaluate the Model**:
   ```bash
   python src/evaluate.py
   ```

4. **Run Inference**:
   ```bash
   python src/inference.py
   ```

## Data

The project uses the [Medical Transcriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) dataset from Kaggle, which contains:

- Medical transcriptions across various specialties
- Descriptions and keywords for each transcription
- Different types of medical documentation

## Models

The project supports language model fine-tuning using LoRA (Low-Rank Adaptation) for efficient parameter tuning:

1. **Google's Gemma 2B**: A lightweight yet powerful model, ideal for efficient fine-tuning on consumer hardware
2. **Meta's LLaMA**: A highly capable model known for strong performance on domain-specific tasks

### Default Model

The default model is specified in `src/config.py` and can be overridden via the `--model_name` CLI argument.

### Fine-Tuning Approach

- **Method**: Supervised Fine-Tuning (SFT) with LoRA adapters
- **Trainer**: HuggingFace `SFTTrainer` from TRL library
- **Quantization**: Support for bfloat16 and float16 precision
- **Adapters**: LoRA configuration for memory-efficient fine-tuning

### Training Libraries

The following libraries are used for efficient training:

- `transformers` for loading models and tokenizers
- `peft` for LoRA adapter configuration
- `trl` for `SFTTrainer` and `SFTConfig`
- `datasets` for dataset loading and processing
- `torch` for PyTorch framework

## License

This project is licensed under the terms of the LICENSE file included in the repository.
