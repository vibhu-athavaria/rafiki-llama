import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# --- TEMP DIR FIX for RunPod ---
os.environ["TMPDIR"] = "/tmp"
os.environ["HF_HOME"] = "/tmp"  # optional: avoids writing cache to /root

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()

def main():
    args = parse_args()

    # --- Load dataset ---
    dataset = load_dataset("json", data_files=args.dataset, split="train")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_length = min(tokenizer.model_max_length, 512)

    # --- Tokenization function for batched=True ---
    def tokenize_fn(batch):
        prompts = [
            f"Instruction: {instr}\nResponse: {out}"
            for instr, out in zip(batch["instruction"], batch["output"])
        ]
        tokens = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names
    )

    # --- BitsAndBytesConfig for 4-bit/8-bit loading ---
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="fp4",  # can also be "nf4"
        bnb_8bit_compute_dtype=torch.bfloat16
    )

    # --- Load model ---
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=bnb_config
    )
    model.config.use_cache = False  # Needed for gradient checkpointing with LoRA

    # --- LoRA config ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # --- Training args ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_dir="./logs",
        logging_steps=50,
        save_strategy="steps",
        save_steps=1000,
        fp16=False,
        bf16=True,
        eval_strategy="no",  # No evaluation
        gradient_checkpointing=True
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    trainer.train()

    # --- Save ---
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
