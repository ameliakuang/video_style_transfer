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

from extract_latents import get_latents
from policy_network import LatentPolicy
from style_env import StyleEnv

def evaluate_policy(policy, content_latents, style_latents, content_file, style_file, prev_modified_stylized_frame, prev_ori_stylized_frame, pnp, scheduler, device, save_dir=None):
    """
    Evaluate the policy on a single content-style pair
    Returns:
        Dictionary containing all evaluation metrics
    """
    style_env = StyleEnv(pnp, scheduler, device, content_latents, style_latents, 
                        content_file, style_file, content_latents, content_file)
    
    with torch.no_grad():
        delta_z, _ = policy.sample(content_latents[-1][None, :, :, :], content_latents[-1][None, :, :, :])
        reward, content_loss, style_loss, temporal_loss, loss_modified, _, _, _, _, stylized_frame, ori_stylized_frame = style_env.step(delta_z, prev_modified_stylized_frame, prev_ori_stylized_frame)
    
    if save_dir:
        save_path = os.path.join(save_dir, f'eval_frame_{os.path.basename(content_file)}_{os.path.basename(style_file)}')
        stylized_frame.save(save_path)
    
    # Calculate additional evaluation metrics
    # reference_frame = Image.open(content_file).convert("RGB")
    # clip_score = calculate_clip_score(stylized_frame, reference_frame)
    # stylized_ori = Image.open(prev_modified_stylized_frame).convert("RGB")
    # clip_score_ori = calculate_clip_score(stylized_ori, reference_frame)
    # fvd_score = calculate_fvd_score(stylized_frame)

    is_score = 0
    fvd_score = 0
    
    metrics = {
        'reward': reward,
        'content_loss': content_loss,
        'style_loss': style_loss,
        'temporal_loss': temporal_loss,
        'total_loss': loss_modified,
        # 'is_score': is_score,
        # 'fvd_score': fvd_score
    }
    
    return metrics, stylized_frame, ori_stylized_frame

def plot_metrics(train_metrics, train_eval_metrics, test_eval_metrics, save_dir):
    """
    Plot training and evaluation metrics
    Args:
        train_metrics: Dictionary containing lists of training metrics
        train_eval_metrics: Dictionary containing lists of evaluation metrics on training set
        test_eval_metrics: Dictionary containing lists of evaluation metrics on test set
    """
    plt.figure(figsize=(20, 15))

    # Plot training metrics
    plt.subplot(3, 2, 1)
    plt.plot(train_metrics['epoch'], train_metrics['reward'], label='Training Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Training Reward')
    plt.legend()
    plt.grid(True)

    # Plot training vs evaluation rewards
    plt.subplot(3, 2, 2)
    plt.plot(train_metrics['epoch'], train_metrics['reward'], label='Training', color='blue')
    plt.plot(train_eval_metrics['epoch'], train_eval_metrics['reward'], label='Train Eval', color='green', marker='o')
    plt.plot(test_eval_metrics['epoch'], test_eval_metrics['reward'], label='Test Eval', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Training vs Evaluation Rewards')
    plt.legend()
    plt.grid(True)

    # Plot evaluation metrics
    metric_names = ['content_loss', 'style_loss', 'temporal_loss', 'total_loss']
    for idx, metric in enumerate(metric_names):
        plt.subplot(3, 2, idx + 3)
        plt.plot(train_eval_metrics['epoch'], train_eval_metrics[metric], label='Train Eval', marker='o')
        plt.plot(test_eval_metrics['epoch'], test_eval_metrics[metric], label='Test Eval', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} over Time')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_metrics.png'))
    plt.close()

def run_policy_gradients(train_content_paths, train_content_latents,
                        test_content_paths, test_content_latents,
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

    policy = LatentPolicy().to(opt.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=opt.lr)

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
    
    best_reward = float('-inf')
    eval_interval = max(1, num_epochs // 2)

    # Training loop
    for e in range(num_epochs):
        print(f"Epoch {e + 1}/{num_epochs}")
        epoch_metrics = {key: [] for key in train_metrics.keys()}
        epoch_modified_stylized_frames = []
        epoch_ori_stylized_frames = []

        # Training
        for i, (curr_content_latents, curr_content_file) in enumerate(zip(train_content_latents, train_content_paths)):
            if i == 0:
                continue
            prev_content_latents = train_content_latents[i-1]
            prev_content_file = train_content_paths[i-1]

            for j, (style_latents, style_file) in enumerate(zip(all_style_latents, style_paths)):
                style_env = StyleEnv(pnp, scheduler, opt.device,
                                    curr_content_latents, style_latents,
                                    curr_content_file, style_file,
                                    prev_content_latents, prev_content_file,
                                    content_weight=content_weight,
                                    temporal_weight=temporal_weight,
                                    style_weight=style_weight)
                
                with autocast():
                    delta_z, log_prob = policy.sample(prev_content_latents[-1][None, :, :, :], 
                                                    curr_content_latents[-1][None, :, :, :])
                    
                    # obtain reward
                    prev_modified_stylized_frame = None
                    prev_ori_stylized_frame = None
                    if epoch_modified_stylized_frames and epoch_ori_stylized_frames:
                        prev_modified_stylized_frame = epoch_modified_stylized_frames[-1]
                        prev_ori_stylized_frame = epoch_ori_stylized_frames[-1]
                    reward, content, style, temporal, loss_modified, content_ori, style_ori, temporal_ori, loss_ori, modified_stylized_frame, ori_stylized_frame = style_env.step(delta_z, prev_modified_stylized_frame, prev_ori_stylized_frame)
                    
                    # update policy
                    policy_loss = -log_prob[-1] * reward
                    policy_loss = policy_loss / opt.accumulation_steps
                scaler.scale(policy_loss).backward()

                if (i * len(style_paths) + j + 1) % opt.accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_metrics['policy_loss'].append(policy_loss.detach().cpu().item())
                epoch_metrics['content_loss'].append(content)
                epoch_metrics['style_loss'].append(style)
                epoch_metrics['temporal_loss'].append(temporal)
                epoch_metrics['total_loss'].append(loss_modified)
                epoch_metrics['reward'].append(reward)

                epoch_modified_stylized_frames.append(modified_stylized_frame)
                epoch_ori_stylized_frames.append(ori_stylized_frame)

        # Average and store training metrics - all metrics should be CPU values now
        for key in train_metrics.keys():
            if key == 'epoch':
                train_metrics[key].append(e)
            else:
                train_metrics[key].append(np.mean(epoch_metrics[key]))

        # Evaluation
        if (e + 1) % eval_interval == 0 or e == num_epochs - 1:
            print(f"\nRunning evaluation at epoch {e + 1}")
            
            # Evaluate on training set
            train_eval_save_dir = os.path.join(train_eval_path, f'epoch_{e+1}')
            os.makedirs(train_eval_save_dir, exist_ok=True)
            
            epoch_train_eval_metrics = {key: [] for key in train_eval_metrics.keys()}
            
            for i, (content_latents, content_file) in enumerate(zip(train_content_latents, train_content_paths)):
                for style_latents, style_file in zip(all_style_latents, style_paths):
                    metrics, modified_stylized_frame, ori_stylized_frame = evaluate_policy(
                        policy, content_latents, style_latents,
                        content_file, style_file, prev_modified_stylized_frame, prev_ori_stylized_frame, pnp, scheduler,
                        opt.device, train_eval_save_dir
                    )
                    
                    for key in metrics.keys():
                        epoch_train_eval_metrics[key].append(metrics[key])
                    
                    epoch_train_eval_metrics['epoch'].append(e)
            
            # Store training evaluation metrics
            for key in train_eval_metrics.keys():
                train_eval_metrics[key].append(np.mean(epoch_train_eval_metrics[key]))
            
            # Evaluate on test set
            test_eval_save_dir = os.path.join(test_eval_path, f'epoch_{e+1}')
            os.makedirs(test_eval_save_dir, exist_ok=True)
            
            epoch_test_eval_metrics = {key: [] for key in test_eval_metrics.keys()}

            epoch_test_modified_stylized_frames = []
            epoch_test_ori_stylized_frames = []
            
            for i, (content_latents, content_file) in enumerate(zip(test_content_latents, test_content_paths)):
                for style_latents, style_file in zip(all_style_latents, style_paths):
                    
                    if i == 0:
                        prev_test_modified_stylized_frame = None
                        prev_test_ori_stylized_frame = None
                    else:
                        prev_test_modified_stylized_frame = epoch_test_modified_stylized_frames[-1]
                        prev_test_ori_stylized_frame = epoch_test_ori_stylized_frames[-1]
                    
                    metrics, modified_stylized_frame, ori_stylized_frame = evaluate_policy(
                        policy, content_latents, style_latents,
                        content_file, style_file, prev_test_modified_stylized_frame, prev_test_ori_stylized_frame, pnp, scheduler,
                        opt.device, test_eval_save_dir
                    )
                    
                    for key in metrics.keys():
                        epoch_test_eval_metrics[key].append(metrics[key])
                    
                    epoch_test_eval_metrics['epoch'].append(e)

                    epoch_test_modified_stylized_frames.append(modified_stylized_frame)
                    epoch_test_ori_stylized_frames.append(ori_stylized_frame)
            
            # baseline_sims = compute_clip_similarities(epoch_test_modified_stylized_frames)
            # rl_sims = compute_clip_similarities(epoch_test_ori_stylized_frames)

            # Store test evaluation metrics
            for key in test_eval_metrics.keys():
                test_eval_metrics[key].append(np.mean(epoch_test_eval_metrics[key]))
            
            # Print evaluation results
            print("\nTraining Set Evaluation:")
            for k, v in train_eval_metrics.items():
                print(f"{k}: {v[-1]:.4f}")
            
            print("\nTest Set Evaluation:")
            for k, v in test_eval_metrics.items():
                print(f"{k}: {v[-1]:.4f}")
            
            # Save best model based on test set reward
            if test_eval_metrics['reward'][-1] > best_reward:
                best_reward = test_eval_metrics['reward'][-1]
                torch.save({
                    'epoch': e + 1,
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'test_metrics': test_eval_metrics,
                }, os.path.join(model_save_path, 'best_policy_model.pth'))
            
            # Save latest model
            torch.save({
                'epoch': e + 1,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'test_metrics': test_eval_metrics,
            }, os.path.join(model_save_path, 'latest_policy_model.pth'))

    # Plot and save metrics
    plot_metrics(train_metrics, train_eval_metrics, test_eval_metrics, metrics_save_path)

    return policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_content_path', type=str, default='images_mini/frames_1171471')
    parser.add_argument('--test_content_path', type=str, default='images_mini/frames_855867')
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
    train_content_paths, train_content_latents, style_paths, style_latents = get_latents(opt, opt.train_content_path, mode="train")
    test_content_paths, test_content_latents, _, _ = get_latents(opt, opt.test_content_path, mode="test")

    # Run training
    trained_policy = run_policy_gradients(
        train_content_paths, train_content_latents,
        test_content_paths, test_content_latents,
        style_paths, style_latents,
        latents_dir=opt.latents_dir,
        output_dir=opt.output_dir,
        num_epochs=opt.num_epochs,
        lr=opt.lr,
        temporal_weight=opt.temporal_weight,
        content_weight=opt.content_weight,
        style_weight=opt.style_weight
    )
