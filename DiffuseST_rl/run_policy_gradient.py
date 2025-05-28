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

from extract_latents import get_latents
from policy_network import LatentPolicy
from style_env import StyleEnv

def run_policy_gradients(opt, content_path, all_content_latents, style_path, all_style_latents):
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

    policy = LatentPolicy().to(opt.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=opt.lr)

    # start RL training loop 
    for e in range(opt.num_epochs):
        print(f"Epoch {e + 1}/{opt.num_epochs}")
        epoch_rewards = []
        epoch_losses = []

        # per epoch i process all frames
        for i, (curr_content_latents, curr_content_file) in enumerate(zip(all_content_latents, content_path)):
            print(f"Processing content file and latents: {curr_content_latents.shape}, {curr_content_file}")
            
            if i < len(all_content_latents) - 1:
                next_content_latents = all_content_latents[i+1]
                next_content_file = content_path[i+1]

                for style_latents, style_file in zip(all_style_latents, style_path):
                    print(f"in run_policy_gradients style_file{style_file}")
                    style_env = StyleEnv(pnp,
                                        scheduler,
                                        opt.device,
                                        curr_content_latents,
                                        style_latents,
                                        curr_content_file,
                                        style_file,
                                        next_content_latents,
                                        next_content_file)
                    # obtain delta_z
                    delta_z, log_prob = policy.sample(curr_content_latents, next_content_latents)
                    print(f"delta_z shape: {delta_z.shape}")
                    # obtain reward
                    reward = style_env.step(delta_z)
                    # update policy
                    policy_loss = -log_prob * reward
                    optimizer.zero_grad()
                    policy_loss.backward()
                    optimizer.step()

                    epoch_rewards.append(reward)
                    

    # trained policy yay
                    










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
    
    # RL-specific arguments
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of policy gradient optimization epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for policy gradient optimization')
    parser.add_argument('--temporal_weight', type=float, default=1.0,
                        help='Weight for temporal consistency loss')
    parser.add_argument('--content_weight', type=float, default=0.5,
                        help='Weight for content preservation loss')
    
    opt = parser.parse_args()

    content_path, all_content_latents, style_path, all_style_latents = get_latents(opt)


    run_policy_gradients(content_path, all_content_latents, style_path, all_style_latents,
           num_epochs=opt.num_epochs,
           lr=opt.lr,
           temporal_weight=opt.temporal_weight,
           content_weight=opt.content_weight)
