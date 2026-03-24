import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from evaluate import load
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Конфигурация
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
TEST_SIZE = 70                     # 70 примеров для теста (как в задании)
OUTPUT_FILE = "results_baseline.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Загрузка метрик
bleu = load("bleu")
rouge = load("rouge")

# Загрузка датасета
print("Loading dataset...")
dataset = load_dataset("linxy/LaTeX_OCR", split="train")
print(f"Total samples: {len(dataset)}")

# Берём первые TEST_SIZE примеров (как тест)
test_dataset = dataset.select(range(TEST_SIZE))

# Для one-shot выберем один пример из train, не входящий в тест (например, индекс 100)
one_shot_idx = 100
one_shot_example = dataset[one_shot_idx]
one_shot_text = f"Example: {one_shot_example['text']}\nOutput: {one_shot_example['text']}"

print(f"One-shot example (idx={one_shot_idx}): {one_shot_example['text'][:80]}...")

# Загрузка модели
print(f"\nLoading model {MODEL_NAME}...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model loaded.")

def predict(image, prompt, one_shot=False):
    if one_shot:
        full_prompt = f"{one_shot_text}\n\n{prompt}"
    else:
        full_prompt = prompt

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": full_prompt}
            ]
        }
    ]
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

results = {"zero_shot": {"predictions": [], "metrics": None},
           "one_shot": {"predictions": [], "metrics": None}}

for shot_type in ["zero_shot", "one_shot"]:
    print(f"\n===== {shot_type.upper()} =====")
    predictions = []
    references = []

    for idx in tqdm(range(TEST_SIZE), desc=f"Processing {shot_type}"):
        sample = test_dataset[idx]
        pred = predict(sample["image"], "Convert this mathematical formula to LaTeX code:", one_shot=(shot_type == "one_shot"))
        predictions.append(pred)
        references.append(sample["text"])
        results[shot_type]["predictions"].append({
            "idx": idx,
            "true": sample["text"],
            "predicted": pred
        })

    # Метрики
    bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])["bleu"]
    rouge_score = rouge.compute(predictions=predictions, references=references)["rougeL"]

    # Character accuracy
    char_acc = []
    for p, r in zip(predictions, references):
        if not p or not r:
            char_acc.append(0.0)
        else:
            matches = sum(1 for pc, rc in zip(p, r) if pc == rc)
            char_acc.append(matches / max(len(p), len(r)))
    avg_char_acc = np.mean(char_acc)

    results[shot_type]["metrics"] = {
        "bleu": bleu_score,
        "rougeL": rouge_score,
        "char_accuracy": avg_char_acc
    }

    print(f"BLEU: {bleu_score:.4f}")
    print(f"ROUGE-L: {rouge_score:.4f}")
    print(f"Char accuracy: {avg_char_acc:.4f}")

# Сохраняем результаты
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {OUTPUT_FILE}")