import os
import re
import requests
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates  # UI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

MODEL_NAME = "maverick707/ai-text-summarizer-t5"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

# Set this in Render's dashboard under Environment > Environment Variables.
# Use the same token shown in your HF "render-deploy" access token (read is enough,
# but the write one you already have will also work).
HF_TOKEN = os.environ.get("HF_TOKEN")

templates = Jinja2Templates(directory=".")


class DialogueInput(BaseModel):
    dialogue: str


def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)      # lines
    text = re.sub(r"\s+", " ", text)       # spaces
    text = re.sub(r"<.*?>", " ", text)     # html tags <p> <h1>
    text = text.strip().lower()
    return text


def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="HF_TOKEN environment variable is not set on the server."
        )

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": dialogue,
        "parameters": {
            "max_length": 150,
            "num_beams": 4,
            "early_stopping": True,
        },
        # First request to a cold model can 503 while HF loads it into memory.
        "options": {"wait_for_model": True},
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Hugging Face API error ({response.status_code}): {response.text}"
        )

    result = response.json()

    # summarization pipeline -> [{"summary_text": "..."}]
    # text2text-generation pipeline -> [{"generated_text": "..."}]
    if isinstance(result, list) and result:
        item = result[0]
        return item.get("summary_text") or item.get("generated_text") or str(item)

    raise HTTPException(status_code=502, detail=f"Unexpected response format: {result}")


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
