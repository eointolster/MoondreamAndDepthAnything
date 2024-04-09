import cv2
import torch
import numpy as np
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import gradio as gr

# Set up model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = 'vitl'  # Choose between 'vits', 'vitb', 'vitl'
depth_model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(DEVICE).eval()

def depth_view_video(video_path):
    # Define transformations
    transform = Compose([
        Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
               resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    margin_width = 50  # Space between original and depth map

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        input_frame = transform({'image': input_frame})['image']
        input_frame = torch.from_numpy(input_frame).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            depth = depth_model(input_frame)
        depth = depth.squeeze().cpu().numpy()
        depth = np.clip((depth - depth.min()) / (depth.max() - depth.min()), 0, 1) * 255.0
        depth = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_MAGMA)
        # Resize depth to match the frame's height
        depth_resized = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Now concatenate using the resized depth map
        combined_frame = np.hstack([frame, np.ones((frame.shape[0], margin_width, 3), dtype=np.uint8) * 255, depth_resized])

        # Yield the combined frame as a numpy array
        yield cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)

    cap.release()

def depth_view_image(img):
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

with gr.Blocks() as demo:
    gr.Markdown("# Depth Estimation")
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload a Video")
            video_output = gr.Image(type="numpy", label="Video Depth View")
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload an Image")
            image_output = gr.Image(type="numpy", label="Image Depth View")

    video_input.change(depth_view_video, inputs=[video_input], outputs=[video_output])
    image_input.change(depth_view_image, inputs=[image_input], outputs=[image_output])

demo.launch(debug=True)