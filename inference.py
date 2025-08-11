import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_PATH = "outputs/rafiki-llama-lora"

def load_model(model_path=MODEL_PATH):
    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("🔄 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16
    )

    print("🔄 Applying LoRA weights...")
    model = PeftModel.from_pretrained(base_model, model_path)

    model.eval()
    return tokenizer, model

def chat(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    tokenizer, model = load_model()

    print("💬 Rafiki-LLaMA is ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        prompt = f"Instruction: {user_input}\nResponse:"
        response = chat(model, tokenizer, prompt)
        print(f"Rafiki: {response}\n")
