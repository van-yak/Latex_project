# app.py
import streamlit as st
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import glob
import os
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Конфигурация
# ------------------------------------------------------------
BASE_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Функция для поиска адаптера
def find_adapter():
    """Ищет последнюю папку qwen_combined_*/final_adapter или */final_adapter"""
    adapter_dirs = glob.glob("./qwen_combined_*/final_adapter")
    if not adapter_dirs:
        adapter_dirs = glob.glob("./*/final_adapter")
    if not adapter_dirs:
        return None
    # Берём самую новую по дате модификации
    return max(adapter_dirs, key=os.path.getmtime)

# Загрузка модели и процессора с кэшированием
@st.cache_resource
def load_model(adapter_path, use_8bit=False):
    """Загружает базовую модель, процессор и LoRA адаптер"""
    st.info("Загрузка базовой модели...")
    # Если нужно 8-бит – добавляем quantization_config
    if use_8bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)

    st.info("Загрузка LoRA адаптера...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Опционально: объединить веса адаптера с базовой моделью для ускорения инференса
    # model = model.merge_and_unload()

    model.eval()
    st.success("Модель готова!")
    return model, processor

# Функция предсказания
def predict(model, processor, image, debug=False):
    """Возвращает распознанный LaTeX код"""
    # Приводим изображение к RGB (на всякий случай)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    prompt = "Convert this mathematical formula to LaTeX code:"
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Обработка входа
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(DEVICE)

    # Генерация (используем те же параметры, что в успешном eval-скрипте)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # Отсекаем входную часть
    input_len = inputs.input_ids.shape[1]
    output_ids = generated_ids[:, input_len:]
    output = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    if debug:
        st.text(f"Длина входных токенов: {input_len}")
        st.text(f"Длина сгенерированных токенов: {generated_ids.shape[1] - input_len}")
        st.text(f"Декодированный вывод: '{output}'")
        if not output:
            st.warning("Модель вернула пустую строку. Возможно, адаптер не загружен или изображение не распознаётся.")

    return output

# ------------------------------------------------------------
# Основной интерфейс
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="LaTeX OCR – Qwen3-VL + LoRA", layout="wide")
    st.title("📷 Распознавание математических формул (LaTeX)")
    st.markdown("Загрузите изображение формулы, и модель вернёт LaTeX-код.")

    # Поиск адаптера
    adapter_path = find_adapter()
    if adapter_path is None:
        st.error("❌ Адаптер не найден! Убедитесь, что в текущей папке есть каталог `qwen_combined_*/final_adapter` или `*/final_adapter`.")
        st.stop()
    else:
        # Показываем путь в sidebar
        st.sidebar.success(f"Адаптер: `{adapter_path}`")

    # Настройки в sidebar
    st.sidebar.header("Настройки")
    use_8bit = st.sidebar.checkbox("Использовать 8-битную загрузку (экономит память)", value=True)
    merge_weights = st.sidebar.checkbox("Объединить веса LoRA с базовой моделью (ускоряет инференс)", value=False)
    debug_mode = st.sidebar.checkbox("Показать отладочную информацию", value=False)

    # Загрузка модели (кэшируется по adapter_path + use_8bit)
    model, processor = load_model(adapter_path, use_8bit)

    # Если выбрано объединение, и модель ещё не объединена
    if merge_weights and not hasattr(model, 'merged') and hasattr(model, 'merge_and_unload'):
        with st.spinner("Объединение весов LoRA..."):
            model = model.merge_and_unload()
            model.merged = True
            st.sidebar.success("Веса объединены!")

    # Загрузка изображения
    uploaded_file = st.file_uploader("Выберите изображение", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Загруженное изображение", use_column_width=True)

        if st.button("Распознать LaTeX"):
            with st.spinner("Обработка..."):
                try:
                    result = predict(model, processor, image, debug=debug_mode)
                    if result:
                        st.success("Результат:")
                        st.code(result, language="latex")
                        # Простая кнопка копирования (через JavaScript)
                        st.markdown(
                            f"""
                            <script>
                            function copyToClipboard() {{
                                navigator.clipboard.writeText(`{result.replace('`', '\\`')}`);
                                alert('LaTeX скопирован!');
                            }}
                            </script>
                            <button onclick="copyToClipboard()">📋 Скопировать LaTeX</button>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.error("Модель не смогла распознать формулу. Попробуйте другое изображение или проверьте адаптер.")
                except Exception as e:
                    st.error(f"Ошибка: {e}")
                    if debug_mode:
                        st.exception(e)

    # Дополнительный блок: информация о модели
    with st.sidebar.expander("ℹ️ Информация о модели"):
        st.write(f"**Базовая модель:** {BASE_MODEL_NAME}")
        st.write(f"**Адаптер:** {adapter_path}")
        st.write(f"**Устройство:** {DEVICE.upper()}")
        st.write(f"**Trainable параметры:** {model.num_parameters(only_trainable=True) if hasattr(model, 'num_parameters') else 'недоступно'}")
        if hasattr(model, 'peft_config'):
            st.write(f"**LoRA r:** {model.peft_config['default'].r}")

if __name__ == "__main__":
    main()