import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

class StyleTransferTrainer:
    def __init__(self, policy, optimizer, pnp, scheduler, device, 
                 content_weight, style_weight, temporal_weight,
                 temp_loss_fn, loss_fn_alex, vgg, accumulation_steps):
        self.policy = policy
        self.optimizer = optimizer
        self.pnp = pnp
        self.scheduler = scheduler
        self.device = device
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.temporal_weight = temporal_weight
        self.temp_loss_fn = temp_loss_fn
        self.loss_fn_alex = loss_fn_alex
        self.vgg = vgg
        self.accumulation_steps = accumulation_steps
        self.scaler = GradScaler()

    def train_epoch(self, train_content_paths_list, train_content_latents_list,
                   style_paths, all_style_latents, save_dir):
        """Train for one epoch"""
        epoch_metrics = {
            'policy_loss': [], 'content_loss': [], 'style_loss': [],
            'temporal_loss': [], 'total_loss': [], 'reward': []
        }
        epoch_modified_stylized_frames = []
        epoch_ori_stylized_frames = []

        for video_idx, (train_content_paths, train_content_latents) in enumerate(zip(train_content_paths_list, train_content_latents_list)):
            for i, curr_content_latents in enumerate(train_content_latents):
                if i == 0:
                    continue
                prev_content_latents = train_content_latents[i-1]
                curr_content_file = train_content_paths[i]
                prev_content_file = train_content_paths[i-1]

                for j, (style_latents, style_file) in enumerate(zip(all_style_latents, style_paths)):
                    metrics = self._train_step(
                        curr_content_latents, style_latents,
                        curr_content_file, style_file,
                        prev_content_latents, prev_content_file,
                        video_idx, epoch_modified_stylized_frames,
                        epoch_ori_stylized_frames, i, j, save_dir
                    )

                    # Update metrics
                    if metrics:
                        for key in epoch_metrics:
                            epoch_metrics[key].append(metrics[key])

        return {k: np.mean(v) for k, v in epoch_metrics.items()}

    def _train_step(self, curr_content_latents, style_latents,
                    curr_content_file, style_file,
                    prev_content_latents, prev_content_file,
                    video_idx, epoch_modified_stylized_frames,
                    epoch_ori_stylized_frames, i, j, save_dir):
        """Single training step"""
        from style_env import StyleEnv

        style_env = StyleEnv(
            self.pnp, self.scheduler, self.device,
            curr_content_latents, style_latents,
            curr_content_file, style_file,
            prev_content_latents, prev_content_file,
            content_weight=self.content_weight,
            temporal_weight=self.temporal_weight,
            style_weight=self.style_weight,
            temp_loss_fn=self.temp_loss_fn,
            loss_fn_alex=self.loss_fn_alex,
            vgg_model=self.vgg,
            mode="train"
        )

        with autocast():
            delta_z, log_prob = self.policy.sample(
                prev_content_latents[-1][None, :, :, :],
                curr_content_latents[-1][None, :, :, :]
            )

            # Get previous frames if available
            prev_modified_stylized_frame = None
            prev_ori_stylized_frame = None
            if epoch_modified_stylized_frames and epoch_ori_stylized_frames:
                prev_modified_stylized_frame = epoch_modified_stylized_frames[-1]
                prev_ori_stylized_frame = epoch_ori_stylized_frames[-1]

            # Get reward and losses
            reward, content, style, temporal, loss_modified, content_ori, style_ori, temporal_ori, loss_ori, modified_stylized_frame, ori_stylized_frame = style_env.step(
                delta_z, prev_modified_stylized_frame, prev_ori_stylized_frame, video_idx
            )

            # Calculate policy loss
            policy_loss = -log_prob[-1] * reward
            policy_loss = policy_loss / self.accumulation_steps

        # Backward pass
        self.scaler.scale(policy_loss).backward()

        # Update if accumulation steps reached
        if (i * len(style_latents) + j + 1) % self.accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            metrics = {
                'policy_loss': policy_loss.detach().cpu().item(),
                'content_loss': content,
                'style_loss': style,
                'temporal_loss': temporal,
                'total_loss': loss_modified,
                'reward': reward
            }
        else:
            metrics = None

        epoch_modified_stylized_frames.append(modified_stylized_frame)
        epoch_ori_stylized_frames.append(ori_stylized_frame)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            modified_stylized_frame.save(os.path.join(save_dir, f"modified_stylized_frame_{i}_{j}.png"))
            ori_stylized_frame.save(os.path.join(save_dir, f"ori_stylized_frame_{i}_{j}.png"))

        return metrics

    def save_checkpoint(self, epoch, metrics, model_path, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': metrics['train_metrics'],
            'test_metrics': metrics['test_eval_metrics'],
        }
        
        filename = 'best_policy_model.pth' if is_best else 'latest_policy_model.pth'
        torch.save(checkpoint, os.path.join(model_path, filename)) 