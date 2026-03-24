import streamlit as st
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import glob
import os

st.set_page_config(page_title="LaTeX OCR", layout="centered")
st.title("LaTeX OCR")

@st.cache_resource
def load_model():
    adapter_dirs = glob.glob("./qwen_combined_*/final_adapter")
    if not adapter_dirs:
        adapter_dirs = glob.glob("./*/final_adapter")
    if not adapter_dirs:
        st.error("No model found")
        return None, None
    adapter_path = max(adapter_dirs, key=os.path.getmtime)
    st.write(f"Model: {adapter_path}")
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    return model, processor

model, processor = load_model()
if model is None:
    st.stop()

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    if st.button("Generate"):
        with st.spinner("Generating..."):
            # Формируем промпт
            prompt = "Convert this mathematical formula to LaTeX code:"
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id
                )
            # Декодируем только сгенерированную часть
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            # Показываем результат
            st.subheader("LaTeX code:")
            st.code(output, language="latex")
            st.latex(output)