import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from evaluate import load
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Конфигурация
# ------------------------------------------------------------
BASE_MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
ADAPTER_PATH = "./smolvlm_finetuned_lora"    # путь к адаптеру (папка с результатами обучения)
TEST_SIZE = 70
OUTPUT_FILE = "results_smolvlm.json"
PLOTS_DIR = "plots"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Загрузка датасета
# ------------------------------------------------------------
print("Loading dataset...")
dataset = load_dataset("linxy/LaTeX_OCR", split="train")
test_dataset = dataset.select(range(TEST_SIZE))
print(f"Test samples: {len(test_dataset)}")

# ------------------------------------------------------------
# Загрузка модели
# ------------------------------------------------------------
print(f"\nLoading base model {BASE_MODEL_NAME}...")
base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
processor.size = {"longest_edge": 1024}   # синхронизация с обучением

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
print("Adapter loaded.")

# ------------------------------------------------------------
# Функция предсказания
# ------------------------------------------------------------
def predict(image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this mathematical formula to LaTeX code:"}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    # Извлекаем ответ ассистента
    if "<|assistant|>" in generated_text:
        output = generated_text.split("<|assistant|>")[-1].strip()
    else:
        output = generated_text.strip()
    return output

# ------------------------------------------------------------
# Вычисление метрик
# ------------------------------------------------------------
bleu = load("bleu")
rouge = load("rouge")

print("\nEvaluating...")
predictions = []
references = []

for idx in tqdm(range(TEST_SIZE), desc="Processing"):
    sample = test_dataset[idx]
    pred = predict(sample["image"])
    predictions.append(pred)
    references.append(sample["text"])

bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])["bleu"]
rouge_score = rouge.compute(predictions=predictions, references=references)["rougeL"]

char_acc = []
for p, r in zip(predictions, references):
    if not p or not r:
        char_acc.append(0.0)
    else:
        matches = sum(1 for pc, rc in zip(p, r) if pc == rc)
        char_acc.append(matches / max(len(p), len(r)))
avg_char_acc = np.mean(char_acc)

results = {
    "metrics": {
        "bleu": bleu_score,
        "rougeL": rouge_score,
        "char_accuracy": avg_char_acc
    },
    "predictions": [{"true": r, "predicted": p} for r, p in zip(references, predictions)]
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nBLEU: {bleu_score:.4f}")
print(f"ROUGE-L: {rouge_score:.4f}")
print(f"Char accuracy: {avg_char_acc:.4f}")
print(f"Results saved to {OUTPUT_FILE}")

# ------------------------------------------------------------
# Построение графиков, если есть лог обучения
# ------------------------------------------------------------
metrics_file = os.path.join(ADAPTER_PATH, "metrics.json")
if os.path.exists(metrics_file):
    print(f"\nLoading training metrics from {metrics_file}")
    with open(metrics_file, "r") as f:
        train_metrics = json.load(f)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # График loss
    if "train_loss" in train_metrics and "eval_loss" in train_metrics:
        plt.figure(figsize=(12, 5))
        steps = train_metrics["step"]
        plt.plot(steps, train_metrics["train_loss"], label="Train Loss")
        eval_steps = steps[:len(train_metrics["eval_loss"])]
        plt.plot(eval_steps, train_metrics["eval_loss"], label="Val Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.grid(True)
        plt.savefig(f"{PLOTS_DIR}/loss_curve.png", dpi=150)
        print(f"Loss curve saved to {PLOTS_DIR}/loss_curve.png")

    # График accuracy (если есть)
    if "eval_accuracy" in train_metrics:
        plt.figure(figsize=(12, 5))
        steps = train_metrics["step"][:len(train_metrics["eval_accuracy"])]
        plt.plot(steps, train_metrics["eval_accuracy"], label="Val Accuracy")
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Validation Accuracy")
        plt.grid(True)
        plt.savefig(f"{PLOTS_DIR}/accuracy_curve.png", dpi=150)
        print(f"Accuracy curve saved to {PLOTS_DIR}/accuracy_curve.png")
else:
    print("\nNo training metrics found. Skipping plots.")
