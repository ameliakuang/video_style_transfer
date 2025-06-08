import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch

def get_all_videos_from_folder(folder_path):
    video_folders = [os.path.join(folder_path, d) for d in sorted(os.listdir(folder_path)) 
                    if os.path.isdir(os.path.join(folder_path, d))]
    return video_folders

def load_all_videos_latents(video_folders, opt, mode="train"):
    from extract_latents import get_latents
    
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

def setup_experiment_folders(opt):
    video_name = os.path.basename(opt.train_content_path.rstrip('/'))
    
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Sanitize lr for folder name
    lr_str = str(opt.lr).replace('.', 'p').replace('-', 'm')
    
    # Build subdirectory name
    subdir = f"exp_{now}_epochs{opt.num_epochs}_lr{lr_str}_video{video_name}"
    
    # Create output directory structure
    output_dir = os.path.join(opt.output_dir, subdir)
    train_path = os.path.join(output_dir, "train")
    train_eval_path = os.path.join(output_dir, "train_eval")
    test_eval_path = os.path.join(output_dir, "test_eval")
    model_save_path = os.path.join(output_dir, "models")
    metrics_save_path = os.path.join(output_dir, "metrics")
    
    # Create directories
    for path in [output_dir, train_path, train_eval_path, test_eval_path, model_save_path, metrics_save_path]:
        os.makedirs(path, exist_ok=True)
        
    return {
        'output_dir': output_dir,
        'train_path': train_path,
        'train_eval_path': train_eval_path,
        'test_eval_path': test_eval_path,
        'model_save_path': model_save_path,
        'metrics_save_path': metrics_save_path
    }

def plot_metrics(train_metrics, train_eval_metrics, test_eval_metrics, save_dir):
    plt.figure(figsize=(20, 15))

    plt.subplot(3, 2, 1)
    plt.plot(train_metrics['epoch'], train_metrics['reward'], label='Training Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Training Reward')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(train_metrics['epoch'], train_metrics['reward'], label='Training', color='blue')
    plt.plot(train_eval_metrics['epoch'], train_eval_metrics['reward'], label='Train Eval', color='green', marker='o')
    plt.plot(test_eval_metrics['epoch'], test_eval_metrics['reward'], label='Test Eval', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Training vs Evaluation Rewards')
    plt.legend()
    plt.grid(True)

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

def save_metrics(metrics_data, metrics_save_path):
    metrics_json_path = os.path.join(metrics_save_path, 'all_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    # # Save as numpy arrays
    # metrics_np_path = os.path.join(metrics_save_path, 'all_metrics.npz')
    # np.savez(metrics_np_path, **metrics_data) 