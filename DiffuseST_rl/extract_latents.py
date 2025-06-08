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
    all_paths = [f for f in Path(path).glob('*')]
    all_paths.sort()
    all_latents = []

    print(all_paths)

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

        save_path = os.path.join(base_save_path, os.path.splitext(os.path.basename(file))[0])
        os.makedirs(save_path, exist_ok=True)
        check_path = os.path.join(save_path, 'noisy_latents_0.pt')

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

def get_latents(opt, content_path=None, mode="train", output_dir=None):
    model_key = "Salesforce/blipdiffusion"
    blip_diffusion_pipe = BLIP.from_pretrained(model_key, torch_dtype=torch.float16).to(opt.device)
    scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")
    scheduler.set_timesteps(opt.ddpm_steps)

    content_path = content_path if content_path is not None else opt.content_path
    extraction_path = "latents_reverse" if opt.extract_reverse else "latents_forward"
    
    if output_dir is None:
        output_dir = opt.latents_dir
    base_save_path = os.path.join(output_dir, extraction_path)
    os.makedirs(base_save_path, exist_ok=True)

    seed_everything(opt.seed)
    model = Preprocess(blip_diffusion_pipe, opt.device, scheduler=scheduler, sd_version=opt.sd_version, hf_key=None)
    pnp = PNP(blip_diffusion_pipe, opt)

    content_paths, content_latents = extract_latents_for_path(
        model, content_path, scheduler, opt, base_save_path, is_style=False
    )

    style_save_path = os.path.join(opt.latents_dir, "style", extraction_path)
    os.makedirs(style_save_path, exist_ok=True)
    style_paths, style_latents = extract_latents_for_path(
        model, opt.style_path, scheduler, opt, style_save_path, is_style=True
    )

    return content_paths, content_latents, style_paths, style_latents

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
