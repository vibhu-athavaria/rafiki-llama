import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/llama-lora")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()

def main():
    args = parse_args()

    # Load dataset
    dataset = load_dataset("json", data_files=args.dataset, split="train")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    max_length = min(tokenizer.model_max_length, 512)  # cap at 512 if model supports more

    def tokenize_fn(example):
        prompt = f"Instruction: {example['instruction']}\nResponse: {example['output']}"
        tokens = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        return tokens

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        load_in_8bit=True
    )

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_dir="./logs",
        logging_steps=50,
        save_strategy="epoch",
        fp16=True,
        evaluation_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"✅ Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
