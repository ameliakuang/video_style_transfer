import argparse
import torch
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import PNDMScheduler
from diffusers.pipelines import BlipDiffusionPipeline
from lpips import LPIPS
from temporal_loss import TemporalConsistencyLossRAFT
from styleloss import VGGFeatures
from policy_network import LatentPolicy
from pnp_style import PNP, BLIP
from utils import (
    get_all_videos_from_folder, load_all_videos_latents,
    setup_experiment_folders, plot_metrics, save_metrics
)
from trainer import StyleTransferTrainer
from evaluator import StyleTransferEvaluator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_content_path', type=str, default='../data2/train_5')
    parser.add_argument('--test_content_path', type=str, default='../data2/test_2')
    parser.add_argument('--style_path', type=str, default='images/style')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--latents_dir', type=str, default='latents/')
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'])
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ddpm_steps', type=int, default=999)
    parser.add_argument('--steps_to_save', type=int, default=1000)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--inversion_prompt', type=str, default='')
    parser.add_argument('--extract-reverse', action='store_true')
    
    # RL-specific arguments
    parser.add_argument('--num_epochs', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temporal_weight', type=float, default=10)
    parser.add_argument('--content_weight', type=float, default=1)
    parser.add_argument('--style_weight', type=float, default=10)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    
    return parser.parse_args()

def setup_models(opt, paths):
    # Setup BLIP diffusion pipeline
    model_key = "Salesforce/blipdiffusion"
    blip_diffusion_pipe = BLIP.from_pretrained(model_key, torch_dtype=torch.float16).to(opt.device)
    scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")
    scheduler.set_timesteps(opt.ddpm_steps)
    
    pnp = PNP(blip_diffusion_pipe, opt, paths)
    
    policy = LatentPolicy().to(opt.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=opt.lr)
    
    temp_loss_fn = TemporalConsistencyLossRAFT(
        small=True,
        loss_type='l1',
        occlusion=True,
        occ_thresh_px=1.0,
        device=opt.device
    )
    loss_fn_alex = LPIPS(net='alex')
    
    feature_layers = [1, 6, 11, 20, 29]
    vgg = VGGFeatures(feature_layers)
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False
        
    return pnp, scheduler, policy, optimizer, temp_loss_fn, loss_fn_alex, vgg

def main():
    opt = parse_args()
    
    paths = setup_experiment_folders(opt)
    
    train_video_folders = get_all_videos_from_folder(opt.train_content_path)
    test_video_folders = get_all_videos_from_folder(opt.test_content_path)
    
    train_content_paths_list, train_content_latents_list, style_paths, all_style_latents = load_all_videos_latents(
        train_video_folders, opt, mode="train"
    )
    test_content_paths_list, test_content_latents_list, _, _ = load_all_videos_latents(
        test_video_folders, opt, mode="test"
    )
    
    pnp, scheduler, policy, optimizer, temp_loss_fn, loss_fn_alex, vgg = setup_models(opt, paths)
    
    trainer = StyleTransferTrainer(
        policy, optimizer, pnp, scheduler, opt.device,
        opt.content_weight, opt.style_weight, opt.temporal_weight,
        temp_loss_fn, loss_fn_alex, vgg, opt.accumulation_steps
    )
    
    evaluator = StyleTransferEvaluator(
        policy, pnp, scheduler, opt.device,
        opt.content_weight, opt.style_weight, opt.temporal_weight,
        temp_loss_fn, loss_fn_alex, vgg
    )
    
    train_metrics = {
        'epoch': [], 'reward': [], 'content_loss': [],
        'style_loss': [], 'temporal_loss': [], 'total_loss': [],
        'policy_loss': []
    }
    train_eval_metrics = {
        'epoch': [], 'reward': [], 'content_loss': [],
        'style_loss': [], 'temporal_loss': [], 'total_loss': []
    }
    test_eval_metrics = {
        'epoch': [], 'reward': [], 'content_loss': [],
        'style_loss': [], 'temporal_loss': [], 'total_loss': []
    }
    
    best_reward = float('-inf')
    eval_interval = max(1, opt.num_epochs // 2)
    
    for epoch in range(opt.num_epochs):
        print(f"\nEpoch {epoch + 1}/{opt.num_epochs}")
        
        # Training
        epoch_metrics = trainer.train_epoch(
            train_content_paths_list, train_content_latents_list,
            style_paths, all_style_latents,
            paths['train_path']
        )
        
        # Update training metrics
        train_metrics['epoch'].append(epoch)
        for key in epoch_metrics:
            if key in train_metrics:
                train_metrics[key].append(epoch_metrics[key])
        
        # Evaluation
        if (epoch + 1) % eval_interval == 0 or epoch == opt.num_epochs - 1:
            print(f"\nRunning evaluation epoch {epoch + 1}...")
            
            # Evaluate on training set
            train_eval = evaluator.evaluate(
                train_content_paths_list, train_content_latents_list,
                style_paths, all_style_latents,
                paths['train_eval_path'], "train"
            )
            
            # Evaluate on test set
            test_eval = evaluator.evaluate(
                test_content_paths_list, test_content_latents_list,
                style_paths, all_style_latents,
                paths['test_eval_path'], "test"
            )
            
            # Update evaluation metrics
            for metrics_dict, eval_results in [
                (train_eval_metrics, train_eval),
                (test_eval_metrics, test_eval)
            ]:
                metrics_dict['epoch'].append(epoch)
                for key in eval_results:
                    if key in metrics_dict:
                        metrics_dict[key].append(eval_results[key])
            
            # Save best model
            if test_eval['reward'] > best_reward:
                best_reward = test_eval['reward']
                trainer.save_checkpoint(
                    epoch,
                    {'train_metrics': train_metrics, 'test_eval_metrics': test_eval_metrics},
                    paths['model_save_path'],
                    is_best=True
                )
            
            # Save latest model
            trainer.save_checkpoint(
                epoch,
                {'train_metrics': train_metrics, 'test_eval_metrics': test_eval_metrics},
                paths['model_save_path'],
                is_best=False
            )
            
            print("\nTraining Set Evaluation:")
            for k, v in train_eval.items():
                print(f"{k}: {v:.4f}")
            print("\nTest Set Evaluation:")
            for k, v in test_eval.items():
                print(f"{k}: {v:.4f}")
    
    metrics_data = {
        'train_metrics': train_metrics,
        'train_eval_metrics': train_eval_metrics,
        'test_eval_metrics': test_eval_metrics
    }
    save_metrics(metrics_data, paths['metrics_save_path'])

    plot_metrics(train_metrics, train_eval_metrics, test_eval_metrics, paths['metrics_save_path'])
    
    return policy

if __name__ == "__main__":
    main() 