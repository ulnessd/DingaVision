import cv2
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Moondream
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", trust_remote_code=True, revision="2025-04-14"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(
    "vikhyatk/moondream2", trust_remote_code=True, revision="2025-04-14"
)

# Open camera
cap = cv2.VideoCapture(0)  # Use 0 or 1 depending on your webcam index

print("Press SPACE to capture frame, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow("Live Feed", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        # Save frame as image
        img_path = "captured.jpg"
        cv2.imwrite(img_path, frame)
        print("Frame captured. Running model...")

        img = Image.open(img_path).convert("RGB")
        caption = model.caption(img)
        answer = model.query(img, "What is shown in this image?")
        print("Caption:", caption["caption"])
        print("Answer:", answer["answer"])

cap.release()
cv2.destroyAllWindows()
