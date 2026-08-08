import re
import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_NAME = "maverick707/ai-text-summarizer-t5"


@st.cache_resource
def load_model():
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_NAME, low_cpu_mem_usage=True
    )
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    model.eval()
    return model, tokenizer


def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()


def summarize_dialogue(dialogue, model, tokenizer):
    dialogue = clean_data(dialogue)
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=2,
            early_stopping=True,
        )
    return tokenizer.decode(targets[0], skip_special_tokens=True)


st.set_page_config(page_title="Text Summarizer", page_icon="📝")
st.title("📝 Text Summarizer")
st.caption("using a fine-tuned T5 model (Hugging Face)")

model, tokenizer = load_model()

dialogue = st.text_area(
    "Write or paste your content below:",
    height=200,
    placeholder="Enter your content here...",
)

if st.button("Summarize", type="primary"):
    if dialogue.strip():
        with st.spinner("Summarizing..."):
            summary = summarize_dialogue(dialogue, model, tokenizer)
        st.subheader("Content Summary")
        st.write(summary)
    else:
        st.warning("Please enter some text first.")
