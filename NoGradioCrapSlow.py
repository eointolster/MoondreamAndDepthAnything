import cv2
import torch
import numpy as np
from torchvision.transforms import Compose
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from moondream import detect_device, LATEST_REVISION
# Import your specific moondream model and any other necessary libraries here

# Setup device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Load models and tokenizer
model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
moondream = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=LATEST_REVISION
).to(device=DEVICE, dtype=DTYPE)
moondream.eval()




# Load depth estimation model
encoder = 'vitl'
depth_model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(DEVICE).eval()

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

def generate_description(image, prompt, model, tokenizer):
    # Convert image to PIL for processing
    pil_image = Image.fromarray(image)
    image_embeds = model.encode_image(pil_image).to(DEVICE)

    # Assuming a synchronous-like interface for answer_question (this part is speculative)
    answer = model.answer_question(image_embeds=image_embeds, question=prompt, tokenizer=tokenizer)
    return answer

prompt = "What is happening in this image?"
frame_skip = 5  # Number of frames to skip
frame_count = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
        # Process frame for depth estimation
            input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            input_frame = transform({'image': input_frame})['image']
            input_frame = torch.from_numpy(input_frame).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                depth = depth_model(input_frame)
            depth = depth.squeeze().cpu().numpy()
            depth_normalized = np.clip((depth - depth.min()) / (depth.max() - depth.min()), 0, 1)
            depth_colormap = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            depth_resized = cv2.resize(depth_colormap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

            combined_frame = np.hstack([frame, depth_resized])
            cv2.imshow('Depth Estimation and Original Frame', combined_frame)

            # Generate description for the frame
            description = generate_description(frame, prompt, moondream, tokenizer)
            print(f"Descriptive text: {description}")
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()