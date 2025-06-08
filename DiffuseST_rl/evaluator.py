import torch
import numpy as np
import os
from PIL import Image
from style_env import StyleEnv
from torchvision import transforms

class StyleTransferEvaluator:
    def __init__(self, policy, pnp, scheduler, device,
                 content_weight, style_weight, temporal_weight,
                 temp_loss_fn, loss_fn_alex, vgg):
        self.policy = policy
        self.pnp = pnp
        self.scheduler = scheduler
        self.device = device
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.temporal_weight = temporal_weight
        self.temp_loss_fn = temp_loss_fn
        self.loss_fn_alex = loss_fn_alex
        self.vgg = vgg

    def evaluate(self, content_paths_list, content_latents_list,
                style_paths, all_style_latents, save_dir=None, prefix=""):
        eval_metrics = {
            'reward': [], 'content_loss': [],
            'style_loss': [], 'temporal_loss': [], 'total_loss': []
        }

        eval_modified_stylized_frames = []
        eval_ori_stylized_frames = []

        for video_idx, (content_paths, content_latents) in enumerate(zip(content_paths_list, content_latents_list)):
            for i, (content_latents, content_file) in enumerate(zip(content_latents, content_paths)):
                if i == 0:
                    prev_modified_stylized_frame = None
                    prev_ori_stylized_frame = None
                else:
                    prev_modified_stylized_frame = eval_modified_stylized_frames[-1]
                    prev_ori_stylized_frame = eval_ori_stylized_frames[-1]

                for style_latents, style_file in zip(all_style_latents, style_paths):
                    metrics, modified_frame, ori_frame = self._evaluate_step(
                        content_latents, style_latents,
                        content_file, style_file,
                        prev_modified_stylized_frame,
                        prev_ori_stylized_frame,
                        video_idx, save_dir, prefix
                    )

                    for key in metrics:
                        eval_metrics[key].append(metrics[key])

                    eval_modified_stylized_frames.append(modified_frame)
                    eval_ori_stylized_frames.append(ori_frame)

        # Average metrics
        return {k: np.mean(v) for k, v in eval_metrics.items()}

    def _evaluate_step(self, content_latents, style_latents,
                      content_file, style_file,
                      prev_modified_stylized_frame,
                      prev_ori_stylized_frame,
                      video_idx, save_dir=None, prefix=""):
        style_env = StyleEnv(
            self.pnp, self.scheduler, self.device,
            content_latents, style_latents,
            content_file, style_file,
            content_latents, content_file,  # Use same content as prev for single frame eval
            content_weight=self.content_weight,
            temporal_weight=self.temporal_weight,
            style_weight=self.style_weight,
            temp_loss_fn=self.temp_loss_fn,
            loss_fn_alex=self.loss_fn_alex,
            vgg_model=self.vgg,
            mode="train_eval" if prefix == "train" else "test_eval"
        )

        with torch.no_grad():
            delta_z, _ = self.policy.sample(
                content_latents[-1][None, :, :, :],
                content_latents[-1][None, :, :, :]
            )

            reward, content, style, temporal, loss_modified, _, _, _, _, modified_stylized_frame, ori_stylized_frame = style_env.step(
                delta_z, prev_modified_stylized_frame,
                prev_ori_stylized_frame, video_idx
            )

        if save_dir:
            # Create experiment-specific subdirectory for each video
            video_save_dir = os.path.join(save_dir, f'video_{video_idx}')
            os.makedirs(video_save_dir, exist_ok=True)
            
            # Create more descriptive filename
            frame_name = os.path.splitext(os.path.basename(content_file))[0]
            style_name = os.path.splitext(os.path.basename(style_file))[0]
            save_path = os.path.join(
                video_save_dir,
                f'frame_{frame_name}_style_{style_name}.png'
            )
            
            # Ensure image is properly normalized before saving
            if isinstance(modified_stylized_frame, torch.Tensor):
                # Normalize tensor to [0, 1] range
                modified_stylized_frame = modified_stylized_frame.clamp(0, 1)
                # Convert to PIL Image
                modified_stylized_frame = transforms.ToPILImage()(modified_stylized_frame.squeeze(0))
            
            modified_stylized_frame.save(save_path)
            print(f"Saved stylized frame to: {save_path}")  # Add logging

        metrics = {
            'reward': reward,
            'content_loss': content,
            'style_loss': style,
            'temporal_loss': temporal,
            'total_loss': loss_modified
        }

        return metrics, modified_stylized_frame, ori_stylized_frame 