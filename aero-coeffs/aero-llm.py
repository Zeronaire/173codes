import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
import torch.distributed as dist

# Step 1: Load and Prepare Data
df = pd.read_csv("aero_small.csv")  # Load your CSV
numeric_cols = ['H', 'V', 'alpha', 'beta', 'CA', 'CN', 'CZ', 'Cll', 'Cnn', 'Cm', 'CD', 'CL']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Format as text prompts (input -> output)
def format_example(row):
    prompt = f"Estimate aerodynamic coefficients for H={row['H']}, V={row['V']}, alpha={row['alpha']}, beta={row['beta']}."
    response = f"CA={row['CA']:.5f}, CN={row['CN']:.5f}, CZ={row['CZ']:.5f}, Cll={row['Cll']:.5f}, " \
               f"Cnn={row['Cnn']:.5f}, Cm={row['Cm']:.5f}, CD={row['CD']:.5f}, CL={row['CL']:.5f}"
    return {"text": f"<s>[INST] {prompt} [/INST] {response} </s>"}

dataset = Dataset.from_pandas(df).map(format_example, remove_columns=df.columns.tolist())
dataset = dataset.train_test_split(test_size=0.1)  # Split 90/10 for train/val


# Step 2: Load Model and Tokenizer
model_name = "models/llama"  # Use Llama-2-7b (requires HF access token if gated)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Fix for Llama
tokenizer.pad_token_id = tokenizer.eos_token_id


model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Apply LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Adapt attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id

# Tokenize dataset
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")
    tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]  # Copy for each example
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch")

# Step 3: Training Setup
training_args = TrainingArguments(
    output_dir="./llama_aero_finetuned",
    num_train_epochs=40,  # Adjust based on dataset size
    per_device_train_batch_size=4,  # Adjust for GPU memory
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,  # Mixed precision for speed
    report_to="none",  # No WandB
    resume_from_checkpoint = "./llama_aero_finetuned/checkpoint-141300" , # Path to your last checkpoint; adjust as needed
    learning_rate=1e-4,
    lr_scheduler_type="cosine"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Step 4: Fine-Tune
trainer.train(resume_from_checkpoint="./llama_aero_finetuned/checkpoint-141300")

# Step 5: Save Model (merge LoRA for easier inference)
merged_model = model.merge_and_unload()  # Merge adapters into base model
merged_model.save_pretrained("./llama_aero_finetuned_merged")
tokenizer.save_pretrained("./llama_aero_finetuned_merged")

# Step 6: Inference Example
if dist.get_rank() == 0:  # Only run on main process
    model = AutoModelForCausalLM.from_pretrained("./llama_aero_finetuned_merged", torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("./llama_aero_finetuned_merged")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # Test prompt (ensure exact match to training format)
    test_prompt = "<s>[INST] Estimate aerodynamic coefficients for H=100, V=7492.13, alpha=0, beta=-90. [/INST]"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)  # To model device
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,  # Deterministic
        temperature=0.0,  # Low for precision
        top_p=1.0,
        num_beams=1  # Greedy search
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded)
    # Post-process to extract coeffs (e.g., parse response string)
    # Example: import re; coeffs = re.findall(r'(\w+)=([\d.-]+)', decoded.split('[/INST]')[1])