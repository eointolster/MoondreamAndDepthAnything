import cv2
import torch
import numpy as np
from torchvision.transforms import Compose
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import concurrent.futures
import time

# Setup device for each model
DEPTH_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MOONDREAM_DEVICE = 'cpu'

# Setup data type for each device
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Load models and tokenizer
model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
moondream = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True
).to(device=MOONDREAM_DEVICE, dtype=DTYPE)
moondream.eval()

# Load depth estimation model
encoder = 'vitl'
depth_model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(DEPTH_DEVICE).eval()

# Define transformations
transform = Compose([
    Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

# Initialize camera
camera_index = 1
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Shared flag for controlling thread execution
keep_running = True

def generate_description(image, prompt, model, tokenizer):
    pil_image = Image.fromarray(image)
    image_embeds = model.encode_image(pil_image).to(MOONDREAM_DEVICE)
    answer = model.answer_question(image_embeds=image_embeds, question=prompt, tokenizer=tokenizer)
    return answer

prompt = "What is happening in this image?"

def capture_and_display_depth():
    global keep_running
    while keep_running:
        ret, frame = cap.read()
        if not ret:
            break
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        input_frame = transform({'image': input_frame})['image']
        input_frame = torch.from_numpy(input_frame).unsqueeze(0).to(DEPTH_DEVICE)
        with torch.no_grad():
            depth = depth_model(input_frame)
        depth = depth.squeeze().cpu().numpy()
        depth_normalized = np.clip((depth - depth.min()) / (depth.max() - depth.min()), 0, 1)
        depth_colormap = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        depth_resized = cv2.resize(depth_colormap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        combined_frame = np.hstack([frame, depth_resized])
        cv2.imshow('Depth Estimation and Original Frame', combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            keep_running = False
            break

def describe_scene():
    global keep_running
    while keep_running:
        try:
            ret, frame = cap.read()
            if not ret:
                break
            description = generate_description(frame, prompt, moondream, tokenizer)
            print(f"Descriptive text: {description}", flush=True)
            time.sleep(2)  # Control the rate of description generation
        except Exception as e:
            print(f"An error occurred: {e}")
            break

# Run both functions in separate threads
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.submit(capture_and_display_depth)
    executor.submit(describe_scene)

cap.release()
cv2.destroyAllWindows()