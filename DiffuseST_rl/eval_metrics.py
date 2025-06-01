from frechet_video_distance import frechet_video_distance as fvd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

def calculate_fvd_score(real_videos, generated_videos):
    # real_videos and generated_videos are tensors of 
    # [NUMBER_OF_VIDEOS, VIDEO_LENGTH, FRAME_WIDTH, FRAME_HEIGHT, 3] with values in 0-255
    result = fvd.calculate_fvd(
        fvd.create_id3_embedding(fvd.preprocess(real_videos, (224, 224))),
        fvd.create_id3_embedding(fvd.preprocess(generated_videos, (224, 224))))
    return result

def calculate_clip_score(stylized_frame, reference_frame):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if isinstance(stylized_frame, np.ndarray):
        stylized_frame = Image.fromarray(stylized_frame.astype('uint8'))
    if isinstance(reference_frame, np.ndarray):
        reference_frame = Image.fromarray(reference_frame.astype('uint8'))
    
    inputs = processor(images=[stylized_frame, reference_frame], return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    
    similarity = torch.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
    return similarity

def compute_clip_similarities(frames):
    sims = []
    for t in range(len(frames) - 1):
        sim = calculate_clip_score(frames[t], frames[t + 1])
        sims.append(sim)
    return sims

