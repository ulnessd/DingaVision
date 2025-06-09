import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Load the model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device.upper()}")

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    revision="2025-04-14"
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    revision="2025-04-14",
)

# Load your image
image = Image.open("golden.jpeg").convert("RGB")

# Generate a caption
cap = model.caption(image, length="normal")
print("Caption:", cap["caption"])

# Ask a question
ans = model.query(image, "What is in the image?")
print("Answer:", ans["answer"])
