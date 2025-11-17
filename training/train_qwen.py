import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# --------------------------
# Load tokenizer & model
# --------------------------
model_name = "Qwen-2.5-7B-Instruct"  # local or HF repo
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Load model in 4-bit for QLoRA (requires bitsandbytes)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# --------------------------
# Set up LoRA config
# --------------------------
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # commonly tuned for Qwen
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# --------------------------
# Load JSONL dataset
# --------------------------
dataset_path = "qwen_training_data.jsonl"

def preprocess(example):
    # concatenate all assistant/user messages
    full_text = ""
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            full_text += f"### Instruction:\n{content}\n"
        elif role == "assistant":
            full_text += f"### Response:\n{content}\n"
    # tokenize
    tokenized = tokenizer(full_text, truncation=True, max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

raw_dataset = load_dataset("json", data_files=dataset_path)["train"]
tokenized_dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names)

# --------------------------
# Data collator
# --------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")

# --------------------------
# Training arguments
# --------------------------
training_args = TrainingArguments(
    output_dir="./qwen_finetuned",
    per_device_train_batch_size=1,      # fits T4
    gradient_accumulation_steps=8,      # effective batch size = 8
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    report_to="none",
    optim="paged_adamw_32bit"
)

# --------------------------
# Initialize Trainer
# --------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# --------------------------
# Start fine-tuning
# --------------------------
trainer.train()
trainer.save_model("./qwen_finetuned")
print("Fine-tuning complete. Model saved at ./qwen_finetuned")
