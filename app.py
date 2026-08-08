import re
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.templating import Jinja2Templates  # UI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

MODEL_NAME = "maverick707/ai-text-summarizer-t5"

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Dynamic quantization: converts the Linear layer weights from float32 to
# int8. This is what actually saves memory (roughly cuts model RAM usage in
# half to a third) with almost no code change and no accuracy loss worth
# mentioning for a resume project.
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
model.eval()

device = torch.device("cpu")  # free-tier hosts never have a GPU anyway
model.to(device)

templates = Jinja2Templates(directory=".")


class DialogueInput(BaseModel):
    dialogue: str


def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text


def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=2,
            early_stopping=True,
        )

    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary


# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )
