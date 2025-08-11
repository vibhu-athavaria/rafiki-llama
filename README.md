# Rafiki-LLaMA: Instruction-Tuned Coaching Model

This repo trains a LLaMA 3 8B model with LoRA for Rafiki's coaching personality.

## 1. Setup

```bash
git clone <your_repo_url>
cd rafiki-llama
conda create -n rafiki-llama python=3.10 -y
conda activate rafiki-llama
pip install -r requirements.txt


## Generating Dataset

You can create a large instruction dataset (up to 400k Q&A) for fine-tuning:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="sk-..."
export RUNPOD_API_URL="https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
export RUNPOD_API_KEY="..."

# Generate dataset using OpenAI GPT-4
python scripts/generate_dataset.py --target-size 400000

# Or generate dataset using RunPod-hosted model
GENERATOR=runpod python scripts/generate_dataset.py --target-size 400000
