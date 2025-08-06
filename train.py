import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

# -------------------------------
# 1. Config
# -------------------------------
model_name = "meta-llama/Llama-3-8b-hf"  # Needs Hugging Face access
dataset_path = "data/sample_dataset.jsonl"
output_dir = "./lora-rafiki"

# -------------------------------
# 2. Load Model (4-bit for low VRAM)
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True
)

# -------------------------------
# 3. Prepare LoRA Config
# -------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# -------------------------------
# 4. Load Dataset
# -------------------------------
dataset = load_dataset("json", data_files=dataset_path)

# -------------------------------
# 5. Training
# -------------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    save_total_limit=2,
    logging_steps=10,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field=None,  # Will automatically use instruction+response
    max_seq_length=512,
    args=training_args,
)

trainer.train()
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("✅ Training complete. LoRA weights saved to", output_dir)
