import json
import matplotlib.pyplot as plt
import numpy as np

output_dir = "./qwen_combined_20260324_063112"  # укажите вашу папку

with open(f"{output_dir}/metrics.json", "r") as f:
    metrics = json.load(f)

steps = metrics["step"]
train_loss = metrics["train_loss"]
eval_loss = metrics.get("eval_loss", [])

plt.figure(figsize=(12, 5))
plt.plot(steps, train_loss, label="Train Loss", linewidth=2)
if eval_loss:
    eval_steps = steps[:len(eval_loss)]
    plt.plot(eval_steps, eval_loss, label="Val Loss", linewidth=2, marker='o', markersize=3)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/training_curves.png", dpi=150)
print(f"Plot saved to {output_dir}/training_curves.png")