import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.config import Config

def run_inference(prompt):
    # Load Base
    base_model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Load Adapter
    model = PeftModel.from_pretrained(base_model, f"{Config.OUTPUT_DIR}/final_adapter")
    
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    response = tokenizer.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return response

if __name__ == "__main__":
    print(run_inference("Explain quantum computing in one sentence."))