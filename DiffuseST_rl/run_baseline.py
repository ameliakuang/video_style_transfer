from transformers import CLIPTextModel, CLIPTokenizer, logging
#from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers import PNDMScheduler
from diffusers.pipelines import BlipDiffusionPipeline
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler


# suppress partial model loading warning
logging.set_verbosity_error()

import os
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from pnp_utils_style import *
import torchvision.transforms as T
from preprocess_style import get_timesteps, Preprocess
from pnp_style import PNP, BLIP
from lpips import LPIPS
from temporal_loss import TemporalConsistencyLossRAFT
from styleloss import gram_matrix, VGGFeatures

from extract_latents import get_latents
from policy_network import LatentPolicy
from style_env import StyleEnv

def get_all_videos_from_folder(folder_path):
    """Returns sorted list of subfolders containing video frames"""
    video_folders = [os.path.join(folder_path, d) for d in sorted(os.listdir(folder_path)) if os.path.isdir(os.path.join(folder_path, d))]
    return video_folders

def load_all_videos_latents(video_folders, opt, mode="train"):
    """Load latents and frame paths for all videos in a list of folders"""
    all_video_paths = []
    all_video_latents = []
    for vf in video_folders:
        print(f"Processing video folder: {vf}")
        video_name = os.path.basename(os.path.normpath(vf))
        video_output_dir = os.path.join(opt.latents_dir, mode, video_name)
        content_paths, content_latents, style_paths, style_latents = get_latents(
            opt, vf, mode=mode, output_dir=video_output_dir
        )
        all_video_paths.append(content_paths)
        all_video_latents.append(content_latents)
    return all_video_paths, all_video_latents, style_paths, style_latents

def evaluate_policy(policy, content_latents, style_latents, content_file, style_file, prev_modified_stylized_frame, prev_ori_stylized_frame, pnp, scheduler, device, video_num, save_dir=None, temp_loss_fn=None, loss_fn_alex=None, vgg=None):
    """
    Evaluate the policy on a single content-style pair
    Returns:
        Dictionary containing all evaluation metrics
    """
    style_env = StyleEnv(pnp, scheduler, device, content_latents, style_latents, 
                        content_file, style_file, content_latents, content_file,
                        content_weight=content_weight, temporal_weight=temporal_weight, style_weight=style_weight,
                        temp_loss_fn=temp_loss_fn, loss_fn_alex=loss_fn_alex, vgg_model=vgg)
    
    style_env = StyleEnv(pnp, scheduler, opt.device,
                                        curr_content_latents, style_latents,
                                        curr_content_file, style_file,
                                        prev_content_latents, prev_content_file,
                                        content_weight=content_weight,
                                        temporal_weight=temporal_weight,
                                        style_weight=style_weight,
                                        temp_loss_fn=temp_loss_fn,
                                        loss_fn_alex=loss_fn_alex,
                                        vgg_model=vgg)
    
    with torch.no_grad():
        delta_z, _ = policy.sample(content_latents[-1][None, :, :, :], content_latents[-1][None, :, :, :])
        reward, content_loss, style_loss, temporal_loss, loss_modified, _, _, _, _, stylized_frame, ori_stylized_frame = style_env.step(delta_z, prev_modified_stylized_frame, prev_ori_stylized_frame, video_num)
    
    if save_dir:
        video_save_dir = os.path.join(save_dir, f'{video_num}')
        os.makedirs(video_save_dir, exist_ok=True)
        save_path = os.path.join(video_save_dir, f'eval_frame_{os.path.splitext(os.path.basename(content_file))[0]}_{os.path.splitext(os.path.basename(style_file))[0]}.png')
        stylized_frame.save(save_path)
    
    # Calculate additional evaluation metrics
    # reference_frame = Image.open(content_file).convert("RGB")
    # clip_score = calculate_clip_score(stylized_frame, reference_frame)
    # stylized_ori = Image.open(prev_modified_stylized_frame).convert("RGB")
    # clip_score_ori = calculate_clip_score(stylized_ori, reference_frame)

    metrics = {
        'reward': reward,
        'content_loss': content_loss,
        'style_loss': style_loss,
        'temporal_loss': temporal_loss,
        'total_loss': loss_modified
    }
    
    return metrics, stylized_frame, ori_stylized_frame


def run_baseline(train_content_paths_list, train_content_latents_list,
                        style_paths, all_style_latents, latents_dir,output_dir,
                        num_epochs, lr, temporal_weight, content_weight, style_weight):
    model_key = "Salesforce/blipdiffusion"
    blip_diffusion_pipe = BLIP.from_pretrained(model_key, torch_dtype=torch.float16).to(opt.device)

    scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")
    scheduler.set_timesteps(opt.ddpm_steps)
    
    # Create organized output directory structure
    train_eval_path = os.path.join(output_dir, "train_eval")
    test_eval_path = os.path.join(output_dir, "test_eval")
    model_save_path = os.path.join(output_dir, "models")
    metrics_save_path = os.path.join(output_dir, "metrics")
    
    for path in [train_eval_path, test_eval_path, model_save_path, metrics_save_path]:
        os.makedirs(path, exist_ok=True)

    pnp = PNP(blip_diffusion_pipe, opt)

    # Create RAFT, LPIPS, vgg model
    temp_loss_fn = TemporalConsistencyLossRAFT(
        small=True,
        loss_type='l1',
        occlusion=True,
        occ_thresh_px=1.0,
        device=opt.device)
    
    loss_fn_alex = LPIPS(net='alex')
    feature_layers = [1, 6, 11, 20, 29]
    vgg = VGGFeatures(feature_layers)
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False
    

    # training metrics
    train_metrics = {
        'epoch': [],
        'reward': [],   
        'content_loss': [],
        'style_loss': [],
        'temporal_loss': [],
        'total_loss': [],
        'policy_loss': [],
        'is_score': [],
        'fvd_score': []
    }
    
    # evaluation metrics
    train_eval_metrics = {
        'epoch': [],
        'reward': [],
        'content_loss': [],
        'style_loss': [],
        'temporal_loss': [],
        'total_loss': [],
        'is_score': [],
        'fvd_score': []
    }

    test_eval_metrics = {
        'epoch': [],
        'reward': [],
        'content_loss': [],
        'style_loss': [],
        'temporal_loss': [],
        'total_loss': [],
        'is_score': [],
        'fvd_score': []
    }

    model = Preprocess(blip_diffusion_pipe, opt.device, scheduler=scheduler, sd_version=opt.sd_version, hf_key=None)
    

    # Training over all videos
    for video_idx, (train_content_paths, train_content_latents) in enumerate(zip(train_content_paths_list, train_content_latents_list)):
        for i, (curr_content_latents, curr_content_file) in enumerate(zip(train_content_latents, train_content_paths)):
            print(f"Processing video {video_idx} frame {i}: {curr_content_file}, {curr_content_latents.shape}")
            for j, (style_latents, style_file) in enumerate(zip(all_style_latents, style_paths)):
                print(f"Processing style {j}: {style_file}, {style_latents.shape}")
                img = pnp.run_pnp(curr_content_latents, style_latents, style_file, video_idx, content_fn=curr_content_file, style_fn=style_file)
                # TODO: add video name to output file for img[0]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_content_path', type=str, default='../data2/train_mini')
    parser.add_argument('--test_content_path', type=str, default='../data2/test_2')
    parser.add_argument('--style_path', type=str, default='images/style')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--latents_dir', type=str, default='latents/', help='Directory to store/load latents')
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'],
                        help="stable diffusion version")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ddpm_steps', type=int, default=999)
    parser.add_argument('--steps_to_save', type=int, default=1000)
    parser.add_argument('--ddim_steps', type=int, default=50)
    
    parser.add_argument('--inversion_prompt', type=str, default='')
    parser.add_argument('--extract-reverse', default=False, action='store_true', help="extract features during the denoising process")
    
    # RL-specific arguments
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of policy gradient optimization epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for policy gradient optimization')
    parser.add_argument('--temporal_weight', type=float, default=10,
                        help='Weight for temporal consistency loss')
    parser.add_argument('--content_weight', type=float, default=1,
                        help='Weight for content preservation loss')
    parser.add_argument('--style_weight', type=float, default=10,
                        help='Weight for style preservation loss')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Number of steps to accumulate gradients before updating the policy network')
    

    
    opt = parser.parse_args()

    # Get video name from train_content_path (or another relevant arg)
    video_name = os.path.basename(opt.train_content_path.rstrip('/'))

    # Get current time
    now = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Sanitize lr for folder name
    lr_str = str(opt.lr).replace('.', 'p').replace('-', 'm')

    # Build subdirectory name
    subdir = f"exp_{now}_epochs{opt.num_epochs}_lr{lr_str}_video{video_name}"

    # Update output_dir to include the subdirectory
    opt.output_dir = os.path.join(opt.output_dir, subdir)
    os.makedirs(opt.output_dir, exist_ok=True)

    latents_dir = opt.latents_dir
    os.makedirs(latents_dir, exist_ok=True)



    # Get train and test data
    train_video_folders = get_all_videos_from_folder(opt.train_content_path)

    # Load latents and frame paths for all videos in train and test folders
    train_content_paths_list, train_content_latents_list, style_paths, all_style_latents = load_all_videos_latents(train_video_folders, opt, mode="train")

    trained_policy = run_baseline(train_content_paths_list, train_content_latents_list,
                        style_paths, all_style_latents,
                        opt.latents_dir, opt.output_dir,
                        opt.num_epochs, opt.lr,
                        opt.temporal_weight, opt.content_weight, opt.style_weight)

    # # Run training
    # trained_policy = run_policy_gradients(
    #     train_content_paths, train_content_latents,
    #     test_content_paths, test_content_latents,
    #     style_paths, style_latents,
    #     latents_dir=opt.latents_dir,
    #     output_dir=opt.output_dir,
    #     num_epochs=opt.num_epochs,
    #     lr=opt.lr,
    #     temporal_weight=opt.temporal_weight,
    #     content_weight=opt.content_weight,
    #     style_weight=opt.style_weight
    # )
    
    

