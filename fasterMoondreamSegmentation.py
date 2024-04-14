import cv2
import numpy as np
import argparse
import torch
from PIL import Image
from moondream import detect_device, LATEST_REVISION
from transformers import AutoTokenizer, AutoModelForCausalLM

# Argument parser for CPU mode
parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

# Device detection
device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

print("Using device:", device)
if device != torch.device("cpu"):
    print("If you run into issues, pass the `--cpu` flag to this script.")

# Moondream model initialization
model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
moondream = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=LATEST_REVISION
).to(device=device, dtype=dtype)
moondream.eval()

# Function to get a description from the moondream model
def answer_question(img, prompt):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    image_embeds = moondream.encode_image(img_pil)
    description = moondream.answer_question(
        image_embeds=image_embeds,
        question=prompt,
        tokenizer=tokenizer
    )
    return description

def put_text_on_image(cv_img, text, position, max_width, font_scale=0.5, color=(255, 255, 255), thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    wrapped_text = []
    words = text.split(' ')

    # Pre-calculate the height of a single line of text to use for line spacing
    sample_text = "Sample"  # Use any sample text to measure text size
    text_size, _ = cv2.getTextSize(sample_text, font, font_scale, thickness)

    # Create wrapped text to fit the specified maximum width
    while words:
        line = ''
        while words and cv2.getTextSize(line + words[0], font, font_scale, thickness)[0][0] < max_width:
            line += words.pop(0) + ' '
        wrapped_text.append(line)

    # Draw each line on the image, adjusting the vertical position
    y = position[1]
    for line in wrapped_text:
        cv2.putText(cv_img, line.strip(), (position[0], y), font, font_scale, color, thickness)
        y += text_size[1] + 5  # Adjust the y position based on the height of the text and add a small gap


# Initialize the camera
cap = cv2.VideoCapture(0)
persistent_descriptions = [""] * 9
current_segment = 0  # Start with the first segment

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    grid_h, grid_w = h // 3, w // 3
    key = cv2.waitKey(1) & 0xFF

    i, j = divmod(current_segment, 3)
    grid_section = frame[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
    description = answer_question(grid_section, "Give a concise image description as few words as possible")
    persistent_descriptions[current_segment] = description

    for k in range(9):
        i, j = divmod(k, 3)
        text_position = (j*grid_w + 5, i*grid_h + 25)
        segment_width = grid_w - 10
        put_text_on_image(frame, persistent_descriptions[k], text_position, segment_width)

    cv2.imshow('Frame', frame)
    current_segment = (current_segment + 1) % 9  # Move to the next segment

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
