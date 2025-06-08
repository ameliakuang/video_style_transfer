from frechet_video_distance import *
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
# import tensorflow as tf
import time
from functools import lru_cache

@lru_cache(maxsize=1)
def get_clip_model_and_processor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def calculate_clip_score(stylized_frame, reference_frame, max_retries=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = get_clip_model_and_processor()
    
    if isinstance(stylized_frame, np.ndarray):
        stylized_frame = Image.fromarray(stylized_frame.astype('uint8'))
    if isinstance(reference_frame, np.ndarray):
        reference_frame = Image.fromarray(reference_frame.astype('uint8'))
    
    for attempt in range(max_retries):
        try:
            inputs = processor(images=[stylized_frame, reference_frame], return_tensors="pt").to(device)
            with torch.no_grad():
                embeddings = model.get_image_features(**inputs)
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            similarity = torch.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
            return similarity
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return 0.0  # Return default value on failure
            time.sleep(1)  # Wait before retrying

def compute_clip_similarities(frames):
    sims = []
    for t in range(len(frames) - 1):
        sim = calculate_clip_score(frames[t], frames[t + 1])
        sims.append(sim)
    return sims

def compute_style_similarities(frames, style_img):
    sims = []
    for frame in frames:
        sim = calculate_clip_score(frame, style_img)
        sims.append(sim)
    return sims

# def calculate_fvd_score(real_videos, generated_videos):
#     import torch

#     from frechet_video_distance import frechet_video_distance

#     NUMBER_OF_VIDEOS = real_videos.shape[0]
#     VIDEO_LENGTH = real_videos.shape[1]
#     PATH_TO_MODEL_WEIGHTS = "./pytorch_i3d_model/models/rgb_imagenet.pt"

#     # Print input ranges
#     print("Real videos range:", real_videos.min(), real_videos.max())
#     print("Generated videos range:", generated_videos.min(), generated_videos.max())

#     # Convert numpy arrays to torch tensors with the correct dtype
#     real_videos = torch.from_numpy(real_videos).type(torch.FloatTensor)
#     generated_videos = torch.from_numpy(generated_videos).type(torch.FloatTensor)

#     # Calculate FVD
#     fvd = frechet_video_distance(real_videos, generated_videos, PATH_TO_MODEL_WEIGHTS)
    
#     return fvd