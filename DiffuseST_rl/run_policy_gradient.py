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

def run_policy_gradients(content_path, all_content_latents, style_path, all_style_latents, num_epochs, lr, temporal_weight, content_weight):
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

    all_epoch_policy_losses = []
    all_epoch_content_losses = []
    all_epoch_style_losses = []
    all_epoch_temporal_losses = []
    all_epoch_total_losses = []
    all_epoch_reward_history = []
    
    # start RL training loop 
    for e in range(num_epochs):
        print(f"Epoch {e + 1}/{num_epochs}")
        epoch_policy_losses = []
        epoch_content_losses = []
        epoch_style_losses = []
        epoch_temporal_losses = []
        epoch_total_losses = []
        epoch_reward_history = []
        epoch_modified_stylized_frames = []
        epoch_ori_stylized_frames = []

        # per epoch i process all frames
        for i, (curr_content_latents, curr_content_file) in enumerate(zip(all_content_latents, content_path)):
            print(f"Processing content file and latents: {curr_content_latents.shape}, {curr_content_file}")
            
            if i == 0:
                continue
            prev_content_latents = all_content_latents[i-1]
            prev_content_file = content_path[i-1]

            for style_latents, style_file in zip(all_style_latents, style_path):
                print(f"in run_policy_gradients style_file{style_file}")
                style_env = StyleEnv(pnp,
                                    scheduler,
                                    opt.device,
                                    curr_content_latents,
                                    style_latents,
                                    curr_content_file,
                                    style_file,
                                    prev_content_latents,
                                    prev_content_file)
                print(f"type(curr_content_latents): {type(curr_content_latents)}")
                print(f"curr_content_latents.size(): {curr_content_latents.size()}")
                print(f"type(prev_content_latents): {type(prev_content_latents)}")
                print(f"prev_content_latents.size(): {prev_content_latents.size()}")
                # obtain delta_z
                delta_z, log_prob = policy.sample(prev_content_latents[-1][None, :, :, :], curr_content_latents[-1][None, :, :, :])
                print(f"type(delta_z): {type(delta_z)}")
                print(f"delta_z.size(): {delta_z.size()}")
                print(f"type(log_prob): {type(log_prob)}")
                print(f"log_prob.size(): {log_prob.size()}")

                # obtain reward
                prev_modified_stylized_frame = None
                prev_ori_stylized_frame = None
                if epoch_modified_stylized_frames and epoch_ori_stylized_frames:
                    prev_modified_stylized_frame = epoch_modified_stylized_frames[-1]
                    prev_ori_stylized_frame = epoch_ori_stylized_frames[-1]
                reward, content, style, temporal, loss_modified, content_ori, style_ori, temporal_ori, loss_ori, modified_stylized_frame, ori_stylized_frame = style_env.step(delta_z, prev_modified_stylized_frame, prev_ori_stylized_frame)
                # update policy
                policy_loss = -log_prob[-1] * reward
                print(f'type(policy_loss): {type(policy_loss)}')
                print(f'policy_loss.size(): {policy_loss.size()}')
                optimizer.zero_grad()
                policy_loss.backward()
                optimizer.step()

                epoch_policy_losses.append(policy_loss)
                epoch_content_losses.append(content)
                epoch_style_losses.append(style)
                epoch_temporal_losses.append(temporal)
                epoch_total_losses.append(loss_modified)
                epoch_reward_history.append(reward)
                epoch_modified_stylized_frames.append(modified_stylized_frame)
                epoch_ori_stylized_frames.append(ori_stylized_frame)

        all_epoch_policy_losses.append(epoch_policy_losses)
        all_epoch_content_losses.append(epoch_content_losses)
        all_epoch_style_losses.append(epoch_style_losses)
        all_epoch_temporal_losses.append(epoch_temporal_losses)
        all_epoch_total_losses.append(epoch_total_losses)
        all_epoch_reward_history.append(epoch_reward_history)

    # trained policy yay
                    










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_path', type=str,
                        default='images_test/frames_855867') #frames_855867 # frames_1778068
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
