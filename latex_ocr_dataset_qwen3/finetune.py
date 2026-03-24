import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Fine-tuning Qwen3-VL with LoRA (90/10 split, 2000 train samples)")
print("="*60)

# 1. Загрузка датасета
print("\n1. Loading dataset...")
full_dataset = load_dataset("linxy/LaTeX_OCR", split="train")
print(f"Total samples: {len(full_dataset)}")

# 2. Разделение 90/10
train_full, test_full = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.1,
    random_state=42
)
print(f"Train (full): {len(train_full)} samples, Test: {len(test_full)} samples")

# 3. Берём подмножество для обучения (2000 примеров)
train_indices = np.random.choice(train_full, size=2000, replace=False)
train_dataset = full_dataset.select(train_indices)
test_dataset = full_dataset.select(test_full)

print(f"Train (used): {len(train_dataset)} samples")
print(f"Test (used): {len(test_dataset)} samples")

# 4. Загрузка модели в 8-bit
print("\n2. Loading model in 8-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

print("Model loaded!")

# 5. LoRA
print("\n3. Configuring LoRA...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 6. Функция collate для батчей
print("\n4. Preparing data collator...")

def collate_fn(batch):
    input_ids_list = []
    attention_mask_list = []
    pixel_values_list = []
    grid_thw_list = []
    mm_token_type_ids_list = []      # <-- добавили
    
    for example in batch:
        image = example["image"]
        latex = example["text"]
        prompt = "Convert this mathematical formula to LaTeX code:"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = text + latex + processor.tokenizer.eos_token
        
        inputs = processor(
            text=full_text,
            images=[image],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        input_ids_list.append(inputs["input_ids"])
        attention_mask_list.append(inputs["attention_mask"])
        pixel_values_list.append(inputs["pixel_values"])
        grid_thw_list.append(inputs["image_grid_thw"])
        mm_token_type_ids_list.append(inputs["mm_token_type_ids"])   # <-- добавили
    
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_mask = torch.cat(attention_mask_list, dim=0)
    pixel_values = torch.cat(pixel_values_list, dim=0)
    image_grid_thw = torch.cat(grid_thw_list, dim=0)
    mm_token_type_ids = torch.cat(mm_token_type_ids_list, dim=0)    # <-- добавили
    labels = input_ids.clone()
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,   # <-- передаём в модель
        "labels": labels
    }

# 7. Настройка обучения
print("\n5. Setting up training...")

training_args = TrainingArguments(
    output_dir="./latex_ocr_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,        # 2 примера за раз
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,        # эффективный batch = 2*4 = 8
    warmup_steps=50,
    logging_steps=10,
    eval_steps=100,
    save_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs",
    report_to="none",
    dataloader_pin_memory=False,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    remove_unused_columns=False,          # важно, чтобы колонки не удалялись
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn,
)

# 8. Запуск обучения
print("\n6. Starting training...")
print("This may take 2–4 hours. Monitor GPU memory with 'nvidia-smi'")
trainer.train()

# 9. Сохранение модели
print("\n7. Saving model...")
model.save_pretrained("./finetuned_model")
processor.save_pretrained("./finetuned_model")

print("\n Model saved to ./finetuned_model")
print("="*60)
print("Fine-tuning completed!")