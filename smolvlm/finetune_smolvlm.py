import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForImageTextToText as AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

print("=" * 60)
print("Fine-tuning SmolVLM-256M with LoRA (90/10 split, 2000 train samples)")
print("=" * 60)

# 1. Загрузка датасета и выбор 2000 примеров
print("\n1. Loading dataset...")
full_dataset = load_dataset("linxy/LaTeX_OCR", split="train")
print(f"Total samples: {len(full_dataset)}")

# Разделение 90/10
train_full, test_full = train_test_split(
    list(range(len(full_dataset))), test_size=0.1, random_state=42
)
print(f"Train (full): {len(train_full)} samples, Test: {len(test_full)} samples")

# Берём 2000 примеров для обучения
train_indices = np.random.choice(train_full, size=2000, replace=False)
train_dataset = full_dataset.select(train_indices)
test_dataset = full_dataset.select(test_full[:70])  # 70 примеров для валидации

print(f"Train (used): {len(train_dataset)} samples")
print(f"Test (used): {len(test_dataset)} samples")

# 2. Загрузка модели в 8-bit
print("\n2. Loading SmolVLM in 8-bit...")
model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
)

model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(model_name)

# Уменьшим разрешение изображений для экономии памяти
# (длинная сторона будет приведена к 1024 пикселям -> до 4 патчей)
processor.size = {"longest_edge": 1024}
print("Model loaded and processor configured.")

# 3. LoRA подготовка
print("\n3. Configuring LoRA...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # модули внимания декодера
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Функция collate с правильным форматированием и маскировкой
def collate_fn(batch):
    texts = []
    images = []
    for example in batch:
        image = example["image"]
        latex = example["text"]
        prompt = "Convert this mathematical formula to LaTeX code:"

        # Диалог: user (изображение + текст) и assistant (LaTeX)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": latex}],
            },
        ]

        # Применяем chat template, но НЕ добавляем токен начала генерации
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
        images.append(image)

    # Токенизируем текст и изображения вместе
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=1024,   # увеличено, чтобы вместить визуальные токены
        return_tensors="pt",
    )

    # Копируем input_ids для создания меток
    labels = inputs["input_ids"].clone()

    # Маскируем все токены до и включая токен ассистента (<|assistant|>)
    assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|assistant|>")
    for i in range(labels.shape[0]):
        assistant_positions = (labels[i] == assistant_token_id).nonzero(as_tuple=True)[0]
        if len(assistant_positions) > 0:
            pos = assistant_positions[0]
            labels[i, : pos + 1] = -100   # маскируем всё до и включая assistant
        else:
            # Если токен не найден (не должно случиться), маскируем всё
            labels[i, :] = -100

    inputs["labels"] = labels
    return inputs

# 5. Настройка аргументов обучения
training_args = TrainingArguments(
    output_dir="./smolvlm_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,   # эффективный батч = 2*4 = 8
    warmup_steps=50,
    logging_steps=10,
    eval_steps=100,
    save_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    learning_rate=2e-4,
    fp16=True,                       # используем fp16 для ускорения
    logging_dir="./logs",
    report_to="none",
    dataloader_pin_memory=False,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn,
)

# 6. Запуск обучения
print("\n4. Starting training...")
trainer.train()

# 7. Сохранение LoRA адаптера
print("\n5. Saving LoRA adapter...")
model.save_pretrained("./smolvlm_finetuned_lora")
processor.save_pretrained("./smolvlm_finetuned_lora")
print(" LoRA adapter saved to ./smolvlm_finetuned_lora")
