from transformers import CLIPTextModel, CLIPTokenizer, logging
#from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers import PNDMScheduler
from diffusers.pipelines import BlipDiffusionPipeline
import torch.nn.functional as F

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


def extract_latents_for_path(model, path, scheduler, opt, base_save_path, is_style=False):
    """
    Extract latents for a given path (either content or style)
    Args:
        model: Preprocess model
        path: Path to process
        scheduler: Scheduler to use
        opt: Options
        base_save_path: Base path to save latents
        is_style: Whether processing style images (affects number of timesteps)
    Returns:
        tuple: (file_paths, latents)
    """
    all_paths = [f for f in Path(path).glob('*')]
    all_latents = []

    for file in all_paths:
        print(f"Processing {'style' if is_style else 'content'} file: {file}")
        
        # Get timesteps
        timesteps_to_save, num_inference_steps = get_timesteps(
            scheduler, num_inference_steps=opt.ddpm_steps,
            strength=1.0,
            device=opt.device
        )

        if is_style:
            timesteps_to_save = timesteps_to_save[-int(opt.ddpm_steps*opt.alpha):]
        elif opt.steps_to_save < opt.ddpm_steps:
            timesteps_to_save = timesteps_to_save[-opt.steps_to_save:]

        # Setup save path
        save_path = os.path.join(base_save_path, os.path.splitext(os.path.basename(file))[0])
        os.makedirs(save_path, exist_ok=True)
        check_path = os.path.join(save_path, 'noisy_latents_0.pt')

        # Extract or load latents
        if not os.path.exists(check_path):
            print(f"No available latents, start extraction for {file}")
            if is_style:
                model.scheduler.set_timesteps(opt.ddpm_steps)
            
            _, latents = model.extract_latents(
                data_path=file,
                num_steps=opt.ddpm_steps,
                save_path=save_path,
                timesteps_to_save=timesteps_to_save,
                inversion_prompt=opt.inversion_prompt,
                extract_reverse=opt.extract_reverse
            )
        else:
            num_steps = int(opt.ddpm_steps * opt.alpha) if is_style else opt.ddpm_steps
            latents = []
            for t in range(num_steps):
                latents_path = os.path.join(save_path, f'noisy_latents_{t}.pt')
                if os.path.exists(latents_path):
                    latents.append(torch.load(latents_path))
            latents = torch.cat(latents, dim=0).to(opt.device)
        
        all_latents.append(latents)

    return all_paths, all_latents

def get_latents(opt, content_path=None, mode="train"):
    """
    Get latents for content and style images
    Args:
        opt: Options object
        content_path: Optional specific content path (if None, uses opt.content_path)
        mode: Either "train" or "test" to organize output directories
    Returns:
        tuple: (content_paths, content_latents, style_paths, style_latents)
    """
    model_key = "Salesforce/blipdiffusion"
    blip_diffusion_pipe = BLIP.from_pretrained(model_key, torch_dtype=torch.float16).to(opt.device)
    scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")
    scheduler.set_timesteps(opt.ddpm_steps)

    # Setup paths and model
    content_path = content_path if content_path is not None else opt.content_path
    extraction_path = "latents_reverse" if opt.extract_reverse else "latents_forward"
    
    # Create mode-specific output directories
    base_save_path = os.path.join(opt.latents_dir, mode, extraction_path)
    os.makedirs(base_save_path, exist_ok=True)

    # Initialize model
    seed_everything(opt.seed)
    model = Preprocess(blip_diffusion_pipe, opt.device, scheduler=scheduler, sd_version=opt.sd_version, hf_key=None)
    pnp = PNP(blip_diffusion_pipe, opt)

    # Extract content latents
    content_paths, content_latents = extract_latents_for_path(
        model, content_path, scheduler, opt, base_save_path, is_style=False
    )

    # Extract style latents - store in common directory
    style_save_path = os.path.join(opt.latents_dir, "style", extraction_path)
    os.makedirs(style_save_path, exist_ok=True)
    style_paths, style_latents = extract_latents_for_path(
        model, opt.style_path, scheduler, opt, style_save_path, is_style=True
    )

    return content_paths, content_latents, style_paths, style_latents


def run_rl(content_path, all_content_latents, style_path, all_style_latents, pnp, preprocess_model,num_ppo_epochs=5):
    # Process frames sequentially
    for i, (content_latents, content_file) in enumerate(zip(all_content_latents, content_path)):
        print(f"Processing content file and latents: {content_latents.shape}, {content_file}")
        
        if i < len(all_content_latents) - 1:
            next_content_latents = all_content_latents[i+1]
            
            # Load original content frames for loss computation
            content_img_tensor = preprocess_model.load_img(content_file)
            next_content_img_tensor = preprocess_model.load_img(content_path[i+1])

            for style_latents, style_file in zip(all_style_latents, style_path):
                print(f"Processing for style: {style_file}")
                
                # Initial style transfer for current and next frame
                curr_styled_frame = pnp.run_pnp(content_latents, style_latents, style_file, 
                                              content_fn=content_file, style_fn=style_file)[0]
                next_styled_frame = pnp.run_pnp(next_content_latents, style_latents, style_file,
                                              content_fn=content_path[i+1], style_fn=style_file)[0]
                




                
                # Convert styled frames to tensors
                curr_styled_tensor = preprocess_model.load_img(curr_styled_frame)
                next_styled_tensor = preprocess_model.load_img(next_styled_frame)
                
                # PPO optimization loop
                best_reward = float('-inf')
                best_latents = content_latents
                
                for epoch in range(num_ppo_epochs):
                    # Optimize latents using PPO
                    modified_latents = ppo_optimizer.optimize_step(
                        content_latents.unsqueeze(0),
                        next_content_latents.unsqueeze(0),
                        curr_styled_tensor,
                        next_styled_tensor,
                        content_tensor
                    )
                    
                    # Generate new frame with modified latents
                    new_styled_frame = pnp.run_pnp(modified_latents.squeeze(0), style_latents, style_file,
                                                 content_fn=content_file, style_fn=style_file)[0]
                    new_styled_tensor = to_tensor(new_styled_frame).unsqueeze(0).to(pnp.device)
                    
                    # Compute reward
                    reward = ppo_optimizer.compute_reward(
                        new_styled_tensor,
                        next_styled_tensor,
                        content_tensor
                    )
                    
                    # Keep track of best result
                    if reward > best_reward:
                        best_reward = reward
                        best_latents = modified_latents.squeeze(0)
                    
                    print(f"Epoch {epoch + 1}/{num_ppo_epochs}, Reward: {reward.item():.4f}")
                
                # Generate final frame with best latents
                final_styled_frame = pnp.run_pnp(best_latents, style_latents, style_file,
                                               content_fn=content_file, style_fn=style_file)[0]
                
                # Save the final result
                final_styled_frame.save(f'{pnp.config.output_dir}/ppo_{os.path.basename(content_file)}+{os.path.basename(style_file)}.png')
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_path', type=str,
                        default='images/video/frames_855867') #frames_855867 # frames_1778068
    parser.add_argument('--style_path', type=str,
                        default='images/style')
    parser.add_argument('--output_dir', type=str, default='output/frames_855867')
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
    
    opt = parser.parse_args()

    content_path, all_content_latents, style_path, all_style_latents = get_latents(opt)
