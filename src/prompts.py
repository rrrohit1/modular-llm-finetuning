def format_instruction(sample):
    """Formats a single data sample into the Qwen chat template."""
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": sample["instruction"]},
        {"role": "assistant", "content": sample["response"]}
    ]
    return messages

from src.config import Config

def format_github_issue(sample):
    issue_content = f"Title: {sample['title']}\n\nBody: {sample['body']}"
    
    # Pull mapping from Config
    target_label = Config.LABEL_MAP.get(sample.get('label'), "Unknown")

    return [
        {
            "role": "system", 
            "content": "You are a GitHub assistant. Classify the following issue into one of these categories: Bug, Feature, or Question."
        },
        {
            "role": "user", 
            "content": f"Categorize this GitHub issue:\n{issue_content}"
        },
        {
            "role": "assistant", 
            "content": target_label
        }
    ]