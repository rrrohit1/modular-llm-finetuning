def format_instruction(sample):
    """Formats a single data sample into the Qwen chat template."""
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": sample["instruction"]},
        {"role": "assistant", "content": sample["response"]}
    ]
    return messages