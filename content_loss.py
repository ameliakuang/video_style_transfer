import argparse
import lpips
import torch
import os
from PIL import Image
import torchvision.transforms as transforms

def load_frames_from_dir(frame_dir):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    frames = []
    for filename in sorted(os.listdir(frame_dir)):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(frame_dir, filename)
            img = Image.open(img_path)
            tensor = transform(img)
            frames.append(tensor)
    return torch.stack(frames)

def get_content_loss(ref_frame, target_frame):
    loss_fn_alex = lpips.LPIPS(net='alex') 
    # loss_fn_vgg = lpips.LPIPS(net='vgg')
    d = loss_fn_alex(ref_frame.unsqueeze(0), target_frame.unsqueeze(0))  
    return d.item()  # Convert tensor to Python float

def get_video_content_loss(ref_frames, target_frames):
    loss = 0
    for ref_frame, target_frame in zip(ref_frames, target_frames):
        loss += get_content_loss(ref_frame, target_frame)
    return loss / len(ref_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_frames", type=str, default="video/frames_1778068")
    parser.add_argument("--target_frames", type=str, default="output/frames_1778068")
    args = parser.parse_args()

    ref_frames = load_frames_from_dir(args.ref_frames)
    target_frames = load_frames_from_dir(args.target_frames)

    loss = get_video_content_loss(ref_frames, target_frames)
    print(f"Content Loss: {loss:.4f}")


