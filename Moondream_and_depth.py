import cv2
import torch
import numpy as np
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import argparse
import re
import time
from moondream import detect_device, LATEST_REVISION
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM
import gradio as gr

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
moondream = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=LATEST_REVISION
).to(device=DEVICE, dtype=DTYPE)
moondream.eval()

# Set up model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = 'vitl'  # Choose between 'vits', 'vitb', 'vitl'
depth_model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(DEVICE).eval()

def depth_view(img):
    # Define transformations
    transform = Compose([
        Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
               resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    margin_width = 50  # Space between original and depth map

    # Process frame
    input_frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) / 255.0
    input_frame = transform({'image': input_frame})['image']
    input_frame = torch.from_numpy(input_frame).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        depth = depth_model(input_frame)
    depth = depth.squeeze().cpu().numpy()
    depth = np.clip((depth - depth.min()) / (depth.max() - depth.min()), 0, 1) * 255.0
    depth = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_MAGMA)
    # Resize depth to match the frame's height
    depth_resized = cv2.resize(depth, (img.width, img.height), interpolation=cv2.INTER_LINEAR)
    # Now concatenate using the resized depth map
    combined_frame = np.hstack([cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), np.ones((img.height, margin_width, 3), dtype=np.uint8) * 255, depth_resized])
    
    # Return the combined frame as a numpy array
    return cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)

def answer_question(img, prompt):
    image_embeds = moondream.encode_image(img)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    thread = Thread(
        target=moondream.answer_question,
        kwargs={
            "image_embeds": image_embeds,
            "question": prompt,
            "tokenizer": tokenizer,
            "streamer": streamer,
        },
    )
    thread.start()
    buffer = ""
    for new_text in streamer:
        clean_text = re.sub("<$|END$", "", new_text)
        buffer += clean_text
        yield buffer.strip("<END")

with gr.Blocks() as demo:
    gr.Markdown("# 🌔 moondream")
    gr.HTML(
        """
        <style type="text/css">
            .md_output p {
                padding-top: 1rem;
                font-size: 1.2rem !important;
            }
        </style>
        """
    )
    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            value="What's going on? Respond with a single sentence.",
            interactive=True,
        )
    with gr.Row():
        img = gr.Image(type="pil", label="Upload an Image", streaming=True)
        depth_img = gr.Image(label="Depth View", streaming=True)
        output = gr.Markdown(elem_classes=["md_output"])
    latest_img = None
    latest_prompt = prompt.value

    @img.change(inputs=[img])
    def img_change(img):
        global latest_img
        latest_img = img

    @prompt.change(inputs=[prompt])
    def prompt_change(prompt):
        global latest_prompt
        latest_prompt = prompt

    @img.change(inputs=[img], outputs=[depth_img])
    def update_depth_view(img):
        return depth_view(img)

    @demo.load(outputs=[output])
    def live_video():
        while True:
            if latest_img is None:
                time.sleep(0.1)
            else:
                for text in answer_question(latest_img, latest_prompt):
                    if len(text) > 0:
                        yield text

demo.queue().launch(share=True, debug=True)