from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

app = FastAPI()

# Serve static files (including index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")

# Load BLIP model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()

# Hold the most recent image and caption
latest_image = None
latest_caption = ""

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    global latest_image, latest_caption
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    latest_image = image

    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    latest_caption = processor.decode(out[0], skip_special_tokens=True)

    return {"caption": latest_caption}

class PromptRequest(BaseModel):
    caption: str
    prompt: str

@app.post("/interpret")
async def interpret(request: PromptRequest):
    if latest_image is None:
        return JSONResponse(content={"response": "No image has been processed yet."}, status_code=400)

    # For BLIP, we'll concatenate the prompt and caption as context
    prompt_text = f"{request.caption} {request.prompt}"
    inputs = processor(images=latest_image, text=prompt_text, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    response = processor.decode(out[0], skip_special_tokens=True)

    return {"response": response}
