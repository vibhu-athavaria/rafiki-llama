from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "meta-llama/Llama-3-8b-hf"
lora_dir = "./lora-rafiki"

# Load tokenizer and model with LoRA
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", load_in_4bit=True)
model = PeftModel.from_pretrained(model, lora_dir)

# Chat function
def chat(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=150)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

# Example
chat("Motivate me to go for a run today.")
