import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from evaluate import load
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Конфигурация
BASE_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
ADAPTER_PATH = "./finetuned_model"   # папка с сохранённым адаптером
TEST_SIZE = 70
OUTPUT_FILE = "results_finetuned.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Метрики
bleu = load("bleu")
rouge = load("rouge")

# Загрузка датасета
print("Loading dataset...")
dataset = load_dataset("linxy/LaTeX_OCR", split="train")
test_dataset = dataset.select(range(TEST_SIZE))

# Загрузка базовой модели
print(f"\nLoading base model {BASE_MODEL_NAME}...")
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)

# Загрузка адаптера LoRA
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
print("Adapter loaded.")

def predict(image, prompt):
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

print("\nEvaluating fine-tuned model...")
predictions = []
references = []

for idx in tqdm(range(TEST_SIZE), desc="Processing"):
    sample = test_dataset[idx]
    pred = predict(sample["image"], "Convert this mathematical formula to LaTeX code:")
    predictions.append(pred)
    references.append(sample["text"])

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