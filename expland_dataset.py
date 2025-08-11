import json
import os
from tqdm import tqdm
from openai import OpenAI

# Load API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

client = OpenAI(api_key=OPENAI_API_KEY)

INPUT_FILE = "data/seed_dataset.jsonl"
OUTPUT_FILE = "data/expanded_dataset.jsonl"
TARGET_SIZE = 400_000

def load_seed_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def generate_variations(prompt, answer, num_variations=5):
    """
    Generates variations of a prompt-answer pair.
    """
    messages = [
        {"role": "system", "content": "You are an AI dataset generator for a professional coaching assistant. Keep tone supportive, professional, and natural."},
        {"role": "user", "content": f"Generate {num_variations} different variations of the following coaching Q&A:\nQ: {prompt}\nA: {answer}"}
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.9
        )
        text = resp.choices[0].message.content.strip()
        # Expecting list format, split if needed
        return [{"instruction": v.split("A:")[0].replace("Q:", "").strip(),
                 "output": v.split("A:")[1].strip()}
                for v in text.split("\n") if "A:" in v]
    except Exception as e:
        print("Error generating variations:", e)
        return []

def main():
    seeds = load_seed_data()
    dataset = seeds.copy()

    pbar = tqdm(total=TARGET_SIZE, initial=len(dataset))
    while len(dataset) < TARGET_SIZE:
        for seed in seeds:
            variations = generate_variations(seed["instruction"], seed["output"])
            dataset.extend(variations)
            pbar.update(len(variations))
            if len(dataset) >= TARGET_SIZE:
                break

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Expanded dataset saved to {OUTPUT_FILE} with {len(dataset)} entries.")

if __name__ == "__main__":
    main()
