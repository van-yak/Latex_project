import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from evaluate import load
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Конфигурация
MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
TEST_SIZE = 70
OUTPUT_FILE = "results_baseline_smolvlm.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Загрузка датасета
print("Loading dataset...")
dataset = load_dataset("linxy/LaTeX_OCR", split="train")
test_dataset = dataset.select(range(TEST_SIZE))

# Выбираем пример для one-shot (индекс 100)
one_shot_idx = 100
one_shot_example = dataset[one_shot_idx]
one_shot_text = one_shot_example["text"]

# Загрузка модели
print(f"\nLoading model {MODEL_NAME}...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
# Для стабильности установим размер изображения (как в обучении)
processor.size = {"longest_edge": 1024}
print("Model loaded.")

def predict(image, one_shot=False):
    """Функция предсказания с учётом chat template"""
    if one_shot:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Convert this mathematical formula to LaTeX code:"}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": one_shot_text}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Convert this mathematical formula to LaTeX code:"}
                ]
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Convert this mathematical formula to LaTeX code:"}
                ]
            }
        ]
    # Применяем chat template без добавления токена начала генерации
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
    
    # Декодируем только сгенерированную часть (после токена ассистента)
    # В данном случае модель сгенерирует продолжение, начиная с последнего сообщения ассистента
    generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    # Извлекаем ответ ассистента (последнюю часть)
    # Упрощённо: берём последний блок после "assistant"
    if "<|assistant|>" in generated_text:
        parts = generated_text.split("<|assistant|>")
        output = parts[-1].strip()
    else:
        output = generated_text.strip()
    return output

# Метрики
bleu = load("bleu")
rouge = load("rouge")

results = {"zero_shot": {"predictions": [], "metrics": None},
           "one_shot": {"predictions": [], "metrics": None}}

for shot_type in ["zero_shot", "one_shot"]:
    print(f"\n===== {shot_type.upper()} =====")
    predictions = []
    references = []
    for idx in tqdm(range(TEST_SIZE), desc=f"Processing {shot_type}"):
        sample = test_dataset[idx]
        pred = predict(sample["image"], one_shot=(shot_type == "one_shot"))
        predictions.append(pred)
        references.append(sample["text"])
        results[shot_type]["predictions"].append({
            "idx": idx,
            "true": sample["text"],
            "predicted": pred
        })
    
    # Вычисление метрик
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
    
    results[shot_type]["metrics"] = {
        "bleu": bleu_score,
        "rougeL": rouge_score,
        "char_accuracy": avg_char_acc
    }
    print(f"BLEU: {bleu_score:.4f}, ROUGE-L: {rouge_score:.4f}, Char acc: {avg_char_acc:.4f}")

# Сохраняем
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {OUTPUT_FILE}")