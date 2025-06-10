from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import io

app = FastAPI()

# Allow frontend access during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load Moondream ===
print("Loading Moondream model...")
moon_model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", trust_remote_code=True, revision="2025-04-14"
).to(device)
moon_tokenizer = AutoTokenizer.from_pretrained(
    "vikhyatk/moondream2", trust_remote_code=True, revision="2025-04-14"
)
print("Moondream model loaded.")

# === Load Phi ===
print("Loading Phi model...")
phi_model_id = "microsoft/phi-2"
phi_tokenizer = AutoTokenizer.from_pretrained(phi_model_id)
phi_model = AutoModelForCausalLM.from_pretrained(
    phi_model_id,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
).to(device)
print("Phi model loaded.")

# === Moondream wrapper ===
class MoondreamWrapper:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def caption(self, image: Image.Image) -> str:
        result = self.model.caption(image)
        return result["caption"]

moon = MoondreamWrapper(moon_model, moon_tokenizer, device)

# === Phi wrapper ===
def interpret_caption_with_phi(caption: str) -> str:
    prompt = f"""
You are a chemistry assistant.

Reagents detected in the image: {caption}

Which **two reagents** can be combined to form a yellow solid? Be specific about the solid formed and the reaction.

Answer:
"""
    inputs = phi_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = phi_model.generate(**inputs, max_new_tokens=200)
    interpretation = phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return interpretation

# === Endpoint ===
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        print(f"[INFO] Received file: {file.filename}")
        caption = moon.caption(image)
        print(f"[MOONDREAM] Caption: {caption}")

        interpretation = interpret_caption_with_phi(caption)
        print(f"[PHI] Interpretation: {interpretation}")

        return {
            "caption": caption,
            "interpretation": interpretation
        }

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {"error": str(e)}
