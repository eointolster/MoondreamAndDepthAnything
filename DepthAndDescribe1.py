import cv2
import torch
import numpy as np
from torchvision.transforms import Compose
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import concurrent.futures
import queue
import time

# Setup device for each model
DEPTH_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Assuming you always want to use GPU for depth if available
MOONDREAM_DEVICE = 'cpu'  # Explicitly setting Moondream to run on CPU

# Setup data type for each device
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Load models and tokenizer
model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
moondream = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True
).to(device=MOONDREAM_DEVICE, dtype=torch.float32)   # Moondream on CPU, using float32 for better compatibility
moondream.eval()

# Load depth estimation model
encoder = 'vitl'
depth_model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(DEPTH_DEVICE).eval()  # Ensure this uses DEPTH_DEVICE

# Define transformations
transform = Compose([
    Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

# Initialize camera
camera_index = 0
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

def wrap_text(text, width):
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        # Check if adding the new word to the current line would exceed the length limit
        if len(' '.join(current_line + [word])) <= width:
            current_line.append(word)
        else:
            # If it does, start a new line
            lines.append(' '.join(current_line))
            current_line = [word]
    # Make sure to add the last line
    lines.append(' '.join(current_line))
    return lines


def generate_description(image, prompt, model, tokenizer):
    # Convert image to PIL for processing
    pil_image = Image.fromarray(image)
    image_embeds = model.encode_image(pil_image).to(MOONDREAM_DEVICE)  # Ensure embedding goes to CPU
    # Assuming a synchronous-like interface for answer_question (this part is speculative)
    answer = model.answer_question(image_embeds=image_embeds, question=prompt, tokenizer=tokenizer)
    return answer

prompt = "What is happening in this image?"

def capture_and_display_depth(description_queue):
    last_description = ""  # Initialize with an empty string or default message
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame for depth estimation
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        input_frame = transform({'image': input_frame})['image']
        input_frame = torch.from_numpy(input_frame).unsqueeze(0).to(DEPTH_DEVICE)  # Ensure frame tensor is on GPU
        with torch.no_grad():
            depth = depth_model(input_frame)
        depth = depth.squeeze().cpu().numpy()  # Move depth data back to CPU for visualization
        depth_normalized = np.clip((depth - depth.min()) / (depth.max() - depth.min()), 0, 1)
        depth_colormap = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        depth_resized = cv2.resize(depth_colormap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        combined_frame = np.hstack([frame, depth_resized])
        
        # Try to get a new description; if not available, keep the last one
        try:
            description = description_queue.get_nowait()
            last_description = description  # Update the last known description
        except queue.Empty:
            description = last_description  # No new description; reuse the last one
        
        # Display the description in the description box
        description_box = np.zeros((300, combined_frame.shape[1], 3), dtype=np.uint8)
        
        description_lines = wrap_text(description, width=80)  # Adjust the width as needed
        y_offset = 30
        for line in description_lines:
            cv2.putText(description_box, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30  # Adjust line spacing as needed
        
        # Combine the frame and description box
        output_frame = np.vstack([combined_frame, description_box])
        
        cv2.imshow('Depth Estimation and Original Frame', output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def describe_scene(description_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
       
        # Generate description for the frame
        description = generate_description(frame, prompt, moondream, tokenizer)
        description_queue.put(description)
       
        # Sleep for a few seconds to control how often descriptions are generated
        time.sleep(2)

# Create a queue to pass descriptions between threads
description_queue = queue.Queue()

# Run both functions in separate threads
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.submit(capture_and_display_depth, description_queue)
    executor.submit(describe_scene, description_queue)

cap.release()
cv2.destroyAllWindows()