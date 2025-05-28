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


def get_latents(opt):
    model_key = "Salesforce/blipdiffusion"

    blip_diffusion_pipe = BLIP.from_pretrained(model_key, torch_dtype=torch.float16).to(opt.device)
    
    scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")
    scheduler.set_timesteps(opt.ddpm_steps)
    content_path = Path(opt.content_path)
    print(f"Content path: {content_path}")
    content_path = [f for f in content_path.glob('*')]
    print(f"Content path: {content_path}")
    style_path = Path(opt.style_path)
    style_path = [f for f in style_path.glob('*')]

    extraction_path = "latents_reverse" if opt.extract_reverse else "latents_forward"
    base_save_path = os.path.join(opt.output_dir, extraction_path)
    os.makedirs(base_save_path, exist_ok=True)

    pnp = PNP(blip_diffusion_pipe, opt)

    all_content_latents = []
    for content_file in content_path:
        print(f"Processing content file: {content_file}")
        timesteps_to_save, num_inference_steps = get_timesteps(
            scheduler, num_inference_steps=opt.ddpm_steps,
            strength=1.0,
            device=opt.device
        )

        seed_everything(opt.seed)
        if opt.steps_to_save < opt.ddpm_steps:
            timesteps_to_save = timesteps_to_save[-opt.steps_to_save:]

        model = Preprocess(blip_diffusion_pipe, opt.device, scheduler=scheduler, sd_version=opt.sd_version, hf_key=None)
        
        save_path = os.path.join(base_save_path, os.path.splitext(os.path.basename(content_file))[0])
        os.makedirs(save_path, exist_ok=True)
        check_path = os.path.join(save_path, 'noisy_latents_0.pt')

        if not os.path.exists(check_path):
            print(f"No available latents, start extraction for {content_file}")
            _, _, content_latents = model.extract_latents(
                data_path=content_file,
                num_steps=opt.ddpm_steps,
                save_path=save_path,
                timesteps_to_save=timesteps_to_save,
                inversion_prompt=opt.inversion_prompt,
                extract_reverse=opt.extract_reverse
            )
        else:
            content_latents = []
            for t in range(opt.ddpm_steps):
                latents_path = os.path.join(save_path, f'noisy_latents_{t}.pt')
                if os.path.exists(latents_path):
                    content_latents.append(torch.load(latents_path))
            content_latents = torch.cat(content_latents, dim=0).to(opt.device)
        all_content_latents.append(content_latents)

        all_style_latents = []
        for style_file in style_path:
            
            save_path = os.path.join(base_save_path, os.path.splitext(os.path.basename(style_file))[0])
            os.makedirs(save_path, exist_ok=True)
            check_path = os.path.join(save_path, f'noisy_latents_0.pt')
            if not os.path.exists(check_path):
                print(f"No available latents, start extraction for {style_file}")
                timesteps_to_save = timesteps_to_save[-int(opt.ddpm_steps*opt.alpha):]
                model.scheduler.set_timesteps(opt.ddpm_steps)

                _, _, style_latents = model.extract_latents(
                    data_path=style_file,
                    num_steps=opt.ddpm_steps,
                    save_path=save_path,
                    timesteps_to_save=timesteps_to_save,
                    inversion_prompt=opt.inversion_prompt,
                    extract_reverse=opt.extract_reverse
                )

            else:
                style_latents = []
                for t in range(int(opt.ddpm_steps * opt.alpha)):
                    latents_path = os.path.join(save_path, f'noisy_latents_{t}.pt')
                    if os.path.exists(latents_path):
                        style_latents.append(torch.load(latents_path))
                style_latents = torch.cat(style_latents, dim=0).to(opt.device)
            all_style_latents.append(style_latents)
            
    return content_path, all_content_latents, style_path, all_style_latents
    for i, (content_latents, content_file) in enumerate(zip(all_content_latents, content_path)):
        print(f"Processing content file and latents: {content_latents.shape}, {content_file}")
        
        if i < len(all_content_latents) - 1:
            next_content_latents = all_content_latents[i+1]

        for style_latents, style_file in zip(all_style_latents, style_path):
            pnp.run_pnp(content_latents, style_latents, style_file, content_fn=content_file, style_fn=style_file)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
