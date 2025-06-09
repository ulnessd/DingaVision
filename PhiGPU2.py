from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ----------------------------
# CONFIGURATION
# ----------------------------
model_id = "microsoft/phi-2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)


# ----------------------------
# CHEMISTRY PROMPT BUILDER
# ----------------------------
def make_phi_prompt(caption: str, task: str = "yellow_solid") -> str:
    preamble = (
        "You are a chemistry assistant.\n\n"
        f"Reagents detected in the image: {caption.strip()}\n\n"
    )

    if task == "yellow_solid":
        question = "Which **two reagents** can be combined to form a yellow solid? Be specific about the solid formed and the reaction."
    elif task == "no_reaction":
        question = "Which **two reagents** can be combined without producing a visible reaction?"
    elif task == "precipitate_all":
        question = "List all pairs of reagents that would form a precipitate upon mixing, and name the precipitate."
    else:
        question = task

    return f"{preamble}{question}\n\nAnswer:"


# ----------------------------
# CHEMISTRY ANALYSIS
# ----------------------------
def analyze_chemistry(moondream_caption: str, task: str = "yellow_solid") -> str:
    prompt = make_phi_prompt(moondream_caption, task)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Follow-up")[0].strip()  # trim extras


# ----------------------------
# MOCK MOONDREAM OUTPUT
# ----------------------------
# This is close to the natural phrasing Moondream might produce:
fake_caption = "bottles labeled KI, KCl, AgNO3, and Pb(NO3)2"

# Run analysis
result = analyze_chemistry(fake_caption, task="yellow_solid")

# Output
print("\n=== Interpreted Chemistry ===")
print(result)
