import os
import json
from dotenv import load_dotenv
import requests

# Load environment variables from .env
load_dotenv()

RUNPOD_API_URL = os.getenv("RUNPOD_API_URL")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

# Coaching styles
COACHING_STYLES = [
    'Life Coaching',
    'Business Coaching',
    'Career Coaching',
    'Health & Wellness',
    'Executive Coaching',
    'Relationship Coaching',
    'Performance Coaching'
]

def generate_prompt(style):
    return f"""
You are an expert {style} coach.
Generate 5 realistic Q&A pairs where the user asks a question and you provide an expert answer.
Keep it concise, actionable, and natural.
Output in JSONL format with 'instruction' and 'output' fields.
"""

def call_runpod(prompt):
    payload = {
        "input": {
            "prompt": prompt,
            "max_tokens": 512
        }
    }
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(RUNPOD_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def main():
    dataset = []
    for style in COACHING_STYLES:
        print(f"Generating for: {style}")
        prompt = generate_prompt(style)
        result = call_runpod(prompt)

        # Extract output text depending on your RunPod model's schema
        try:
            output_text = result["output"]["text"]
            # Parse each line as JSON if possible
            for line in output_text.strip().split("\n"):
                try:
                    item = json.loads(line)
                    dataset.append(item)
                except json.JSONDecodeError:
                    continue
        except KeyError:
            print("Unexpected RunPod response:", result)

    os.makedirs("data", exist_ok=True)
    with open("data/sample_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} examples to data/sample_dataset.jsonl")

if __name__ == "__main__":
    main()
