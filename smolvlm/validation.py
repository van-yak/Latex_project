import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Qwen3-VL Fine-tuning on Combined Dataset (LaTeX_OCR + MathWriting)")
print("=" * 60)

# -----------------------------------------------------------------------------
# Конфигурация
# -----------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
TRAIN_SAMPLES_LATEX = 1000      # из LaTeX_OCR
TRAIN_SAMPLES_MATH = 1000       # из MathWriting
VAL_SAMPLES = 70                # для валидации
TEST_SAMPLES = 70               # для финального теста
NUM_EPOCHS = 12                 
BATCH_SIZE = 2
GRAD_ACCUM = 4                  
LEARNING_RATE = 2e-4
WARMUP_STEPS = 50
LOGGING_DIR = "./logs"
OUTPUT_DIR = f"./qwen_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
EARLY_STOPPING_PATIENCE = 3

# -----------------------------------------------------------------------------
# 1. Загрузка датасетов с приведением к единому формату
# -----------------------------------------------------------------------------
print("\n1. Loading datasets...")

# LaTeX_OCR
latex_ocr = load_dataset("linxy/LaTeX_OCR", split="train")
print(f"LaTeX_OCR: {len(latex_ocr)} samples")
print("LaTeX_OCR columns:", latex_ocr.column_names)
# Поле 'text' переименовываем в 'latex_code'
latex_ocr = latex_ocr.rename_column("text", "latex_code")

# MathWriting-human
mathwriting = load_dataset("deepcopy/MathWriting-human", split="train")
print(f"MathWriting: {len(mathwriting)} samples")
print("MathWriting columns:", mathwriting.column_names)

# Определяем поле, содержащее LaTeX
possible_fields = ["latex", "text", "formula", "math", "equation"]
latex_field = None
for field in possible_fields:
    if field in mathwriting.column_names:
        latex_field = field
        break

if latex_field is None:
    raise ValueError(f"No known LaTeX field found in MathWriting. Available columns: {mathwriting.column_names}")

print(f"Using field '{latex_field}' for LaTeX in MathWriting")
mathwriting = mathwriting.rename_column(latex_field, "latex_code")

# Удаляем все лишние колонки, оставляем только 'image' и 'latex_code'
latex_ocr = latex_ocr.remove_columns([col for col in latex_ocr.column_names if col not in ["image", "latex_code"]])
mathwriting = mathwriting.remove_columns([col for col in mathwriting.column_names if col not in ["image", "latex_code"]])

# Фильтруем примеры, где latex_code не пустой и не None
def filter_valid(example):
    val = example.get("latex_code")
    # Проверяем, что значение существует, строка и не пустая
    return val is not None and isinstance(val, str) and val.strip() != ""

latex_ocr = latex_ocr.filter(filter_valid)
mathwriting = mathwriting.filter(filter_valid)

print(f"LaTeX_OCR after filtering: {len(latex_ocr)}")
print(f"MathWriting after filtering: {len(mathwriting)}")

# Берём выборки (используем min, чтобы не выйти за границы)
latex_train = latex_ocr.select(range(min(TRAIN_SAMPLES_LATEX, len(latex_ocr))))
math_train = mathwriting.select(range(min(TRAIN_SAMPLES_MATH, len(mathwriting))))

# Объединяем
combined_train = concatenate_datasets([latex_train, math_train])
print(f"Combined train: {len(combined_train)} samples")

# Тестовая выборка (первые 70 из LaTeX_OCR)
test_dataset = latex_ocr.select(range(min(TEST_SAMPLES, len(latex_ocr))))

# Валидационная выборка (примеры 70-139 из LaTeX_OCR)
val_indices = list(range(70, min(70 + VAL_SAMPLES, len(latex_ocr))))
val_dataset = latex_ocr.select(val_indices)

print(f"Train samples: {len(combined_train)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# -----------------------------------------------------------------------------
# 2. Загрузка модели в 8-bit
# -----------------------------------------------------------------------------
print("\n2. Loading model in 8-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model loaded.")

# -----------------------------------------------------------------------------
# 3. LoRA
# -----------------------------------------------------------------------------
print("\n3. Configuring LoRA...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------------------------------------------------------------
# 4. Функция коллатора с усиленной проверкой
# -----------------------------------------------------------------------------
def collate_fn(batch):
    texts = []
    images = []
    for idx, example in enumerate(batch):
        image = example["image"]
        latex = example.get("latex_code")
        # Пропускаем примеры с None или пустой строкой
        if latex is None or not isinstance(latex, str) or not latex.strip():
            print(f"Skipping sample {idx} because latex is invalid: {latex}")
            continue

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
        texts.append(full_text)
        images.append(image)
    
    if len(texts) == 0:
        # Если батч пуст (не должно быть), вернём None (пропустим)
        return None
    
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

# -----------------------------------------------------------------------------
# 5. Кастомный callback для сбора метрик
# -----------------------------------------------------------------------------
class MetricsCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metrics = {"train_loss": [], "eval_loss": [], "step": []}
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.metrics["train_loss"].append(logs["loss"])
                self.metrics["step"].append(state.global_step)
            if "eval_loss" in logs:
                self.metrics["eval_loss"].append(logs["eval_loss"])
    
    def save(self):
        with open(f"{self.output_dir}/metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

# -----------------------------------------------------------------------------
# 6. Аргументы обучения
# -----------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=GRAD_ACCUM,
    warmup_steps=WARMUP_STEPS,
    logging_steps=10,
    eval_steps=100,
    save_steps=500,
    eval_strategy="steps",
    save_strategy="steps",
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_dir=LOGGING_DIR,
    report_to="none",
    dataloader_pin_memory=False,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
)

# -----------------------------------------------------------------------------
# 7. Callbacks
# -----------------------------------------------------------------------------
metrics_callback = MetricsCallback(OUTPUT_DIR)
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_threshold=0.001,
)

# -----------------------------------------------------------------------------
# 8. Trainer
# -----------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_train,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    callbacks=[metrics_callback, early_stopping],
)

# -----------------------------------------------------------------------------
# 9. Обучение
# -----------------------------------------------------------------------------
print("\n4. Starting training...")
trainer.train()

# -----------------------------------------------------------------------------
# 10. Сохранение модели
# -----------------------------------------------------------------------------
metrics_callback.save()

print("\n5. Saving final model...")
model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
processor.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
print(f"✅ Model saved to {OUTPUT_DIR}/final_adapter")

# -----------------------------------------------------------------------------
# 11. Построение графиков
# -----------------------------------------------------------------------------
print("\n6. Plotting training curves...")
plots_dir = f"{OUTPUT_DIR}/plots"
os.makedirs(plots_dir, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. График loss
axes[0, 0].plot(metrics_callback.metrics["step"], metrics_callback.metrics["train_loss"], 
                label="Train Loss", linewidth=2, color='blue')
if metrics_callback.metrics["eval_loss"]:
    steps_eval = metrics_callback.metrics["step"][:len(metrics_callback.metrics["eval_loss"])]
    axes[0, 0].plot(steps_eval, metrics_callback.metrics["eval_loss"], 
                    label="Val Loss", linewidth=2, color='red', marker='o', markersize=4)
axes[0, 0].set_xlabel("Step")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Training and Validation Loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Сглаженный loss
window = 20
if len(metrics_callback.metrics["train_loss"]) > window:
    smoothed = np.convolve(metrics_callback.metrics["train_loss"], 
                           np.ones(window)/window, mode='valid')
    steps_smoothed = metrics_callback.metrics["step"][window-1:]
    axes[0, 1].plot(steps_smoothed, smoothed, label="Smoothed Train Loss", linewidth=2, color='green')
axes[0, 1].set_xlabel("Step")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].set_title(f"Smoothed Loss (window={window})")
axes[0, 1].grid(True, alpha=0.3)

# 3. Loss по эпохам
steps_per_epoch = (len(combined_train) // (BATCH_SIZE * GRAD_ACCUM))
epochs = [i+1 for i in range(len(metrics_callback.metrics["train_loss"]) // steps_per_epoch + 1)]
epoch_losses = []
for i in range(len(epochs)):
    start = i * steps_per_epoch
    end = min((i+1) * steps_per_epoch, len(metrics_callback.metrics["train_loss"]))
    if start < end:
        epoch_losses.append(np.mean(metrics_callback.metrics["train_loss"][start:end]))

axes[1, 0].plot(epochs[:len(epoch_losses)], epoch_losses, marker='o', linewidth=2, color='purple')
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Average Loss")
axes[1, 0].set_title("Loss per Epoch")
axes[1, 0].grid(True, alpha=0.3)

# 4. Разница train-val loss
if metrics_callback.metrics["eval_loss"]:
    min_len = min(len(metrics_callback.metrics["train_loss"][:len(metrics_callback.metrics["eval_loss"])]), 
                  len(metrics_callback.metrics["eval_loss"]))
    if min_len > 0:
        diff = np.array(metrics_callback.metrics["train_loss"][:min_len]) - np.array(metrics_callback.metrics["eval_loss"][:min_len])
        steps_diff = metrics_callback.metrics["step"][:min_len]
        axes[1, 1].plot(steps_diff, diff, linewidth=2, color='orange')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Train - Val Loss")
        axes[1, 1].set_title("Overfitting Indicator (positive = overfitting)")
        axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{plots_dir}/training_curves.png", dpi=150, bbox_inches='tight')
plt.savefig(f"{plots_dir}/training_curves.pdf", bbox_inches='tight')
print(f"✅ Plots saved to {plots_dir}/training_curves.png")

print("\n" + "="*60)
print("Training completed!")
print(f"Model: {OUTPUT_DIR}/final_adapter")
print(f"Metrics: {OUTPUT_DIR}/metrics.json")
print(f"Plots: {plots_dir}/training_curves.png")
print("="*60)