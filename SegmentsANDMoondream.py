import cv2
import numpy as np
import argparse
import torch
import re
from moondream import detect_device, LATEST_REVISION
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM
from PIL import Image

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

def display_grid(frame, rows, cols):
    # Calculate dimensions for each part of the grid
    h, w = frame.shape[:2]
    grid_h, grid_w = h // rows, w // cols
    for i in range(rows):
        for j in range(cols):
            # Extract each grid section
            grid_section = frame[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            # Create a window for each grid section
            window_name = f"Grid_{i+1}_{j+1}"
            cv2.imshow(window_name, grid_section)

# Moondream model initialization
model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
moondream = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=LATEST_REVISION
).to(device=device, dtype=dtype)
moondream.eval()

# Function to get a description from the moondream model
def answer_question(img, prompt, index, descriptions):
    # Convert the NumPy array to a PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
   
    # Now, you can pass this PIL image to the Moondream model
    image_embeds = moondream.encode_image(img_pil)
   
    # Get the description synchronously
    description = moondream.answer_question(
        image_embeds=image_embeds,
        question=prompt,
        tokenizer=tokenizer
    )
   
    descriptions[index] = description

# Function to overlay text on an image
def put_text_on_image(cv_img, text, position, font_scale=1, color=(255, 255, 255)):
    """ Puts text on an image at the given position """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(cv_img, text, position, font, font_scale, color, 2)

# Initialize the camera
cap = cv2.VideoCapture(1)  # Assume 1 is your correct camera index
remove_red_active = False
remove_green_active = False
remove_blue_active = False
edge_detection_active = False
gaussian_blur_active = False
motion_detection_active = False
_, frame1 = cap.read()
_, frame2 = cap.read()

descriptions = [""] * 9  # Initialize with empty descriptions

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calculate dimensions for each part of the grid right here
    h, w = frame.shape[:2]
    grid_h, grid_w = h // 3, w // 3
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        remove_green_active = not remove_green_active
    elif key == ord('2'):
        remove_blue_active = not remove_blue_active
    elif key == ord('3'):
        remove_red_active = not remove_red_active
    elif key == ord('4'):
        edge_detection_active = not edge_detection_active
        gaussian_blur_active = False
    elif key == ord('5'):
        gaussian_blur_active = not gaussian_blur_active
        edge_detection_active = False
    elif key == ord('6'):
        motion_detection_active = not motion_detection_active
    
    if key == ord('7'):
        # Split the frame into 9 segments and process each for description
        threads = []
        for i in range(3):
            for j in range(3):
                index = i * 3 + j
                # Extract each grid section
                grid_section = frame[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                # Start a new thread for each grid section
                thread = Thread(target=answer_question, args=(grid_section, "Give image description", index, descriptions))
                thread.start()
                threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
    
    # Overlay the descriptions on the grid sections
    for i in range(3):
        for j in range(3):
            index = i * 3 + j
            text_position = (j*grid_w + 5, i*grid_h + 25)
            put_text_on_image(frame, descriptions[index], text_position, font_scale=0.5)
    
    if remove_green_active:
        frame[:, :, 1] = 0
    if remove_blue_active:
        frame[:, :, 0] = 0
    if remove_red_active:
        frame[:, :, 2] = 0
    if edge_detection_active:
        frame = cv2.Canny(frame, 100, 200)
    elif gaussian_blur_active:
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
    if motion_detection_active:
        # Process for motion detection
        pass
    
    # Display grid
    display_grid(frame, 3, 3)  # Display the frame as a 3x3 grid
    
    # Display the frame (with the descriptions)
    cv2.imshow('Frame', frame)
    
    if key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()