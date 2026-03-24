import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from evaluate import load
from tqdm import tqdm
import glob
import os
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Настройки
# ------------------------------------------------------------
BASE_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
TEST_SIZE = 70
# Ищем последнюю папку с адаптером, содержащую "qwen_combined" и "final_adapter"
adapter_dirs = glob.glob("./qwen_combined_*/final_adapter")
if not adapter_dirs:
    # Если не нашли, возможно, папка называется по-другому
    adapter_dirs = glob.glob("./*/final_adapter")
if not adapter_dirs:
    raise FileNotFoundError("No adapter directory found. Please specify ADAPTER_PATH manually.")

# Берём самую новую (по дате создания)
ADAPTER_PATH = max(adapter_dirs, key=os.path.getmtime)
print(f"Using adapter: {ADAPTER_PATH}")

OUTPUT_FILE = "results_combined.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Загрузка тестового датасета (первые 70 примеров)
print("Loading dataset...")
dataset = load_dataset("linxy/LaTeX_OCR", split="train")
test_dataset = dataset.select(range(TEST_SIZE))
print(f"Test samples: {len(test_dataset)}")

# Загрузка базовой модели
print(f"\nLoading base model {BASE_MODEL_NAME}...")
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)

# Загрузка адаптера
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
print("Adapter loaded.")

# Функция предсказания
def predict(image):
    prompt = "Convert this mathematical formula to LaTeX code:"
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    return output.strip()

# Метрики
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

# Вычисление метрик
bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])["bleu"]
rouge_score = rouge.compute(predictions=predictions, references=references)["rougeL"]
char_acc = np.mean([
    sum(1 for pc, rc in zip(p, r) if pc == rc) / max(len(p), len(r)) if p and r else 0
    for p, r in zip(predictions, references)
])

results = {
    "metrics": {
        "bleu": bleu_score,
        "rougeL": rouge_score,
        "char_accuracy": char_acc
    },
    "predictions": [{"true": r, "predicted": p} for r, p in zip(references, predictions)]
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nBLEU: {bleu_score:.4f}")
print(f"ROUGE-L: {rouge_score:.4f}")
print(f"Char accuracy: {char_acc:.4f}")
print(f"Results saved to {OUTPUT_FILE}")
