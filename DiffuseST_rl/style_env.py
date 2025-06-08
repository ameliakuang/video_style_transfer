import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from lpips import LPIPS
from temporal_loss import TemporalConsistencyLossRAFT
from styleloss import gram_matrix, VGGFeatures
import torch.nn as nn

class StyleEnv:
    def __init__(self, pnp, 
                 scheduler, 
                 device,
                 curr_content_latents, #[T, C, H, W]
                 style_latents,
                 curr_content_file,
                 style_file,
                 prev_content_latents,
                 prev_content_file,
                 content_weight=1.0,
                 temporal_weight=10.0,
                 style_weight=10.0,
                 temp_loss_fn=None,
                 loss_fn_alex=None,
                 vgg_model=None,
                 mode="train",
                 ):
        self.pnp = pnp
        self.scheduler = scheduler
        self.device = device

        self.curr_content_latents = curr_content_latents
        self.style_latents = style_latents
        self.curr_content_file = curr_content_file
        self.style_file = style_file
        self.prev_content_latents = prev_content_latents
        self.prev_content_file = prev_content_file

        # obtain output file names for debugging purposes - can be removed later
        self.output_modified_fn = f'{self.curr_content_file}_modified'
        self.output_ori_fn = f'{self.curr_content_file}_ori'
        
        self.content_weight = content_weight
        self.temporal_weight = temporal_weight
        self.style_weight = style_weight

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),  # scales to [0, 1]
        ])
        self.curr_content_frame = self.transform(Image.open(self.curr_content_file))
        
        self.temp_loss_fn = temp_loss_fn
        self.loss_fn_alex = loss_fn_alex
        self.vgg_model = vgg_model

        self.mode = mode

    
    def _compute_temporal_loss(self, 
                               stylized_frame,
                               prev_stylized_frame):
        if prev_stylized_frame is None:
            return 0.0

        curr_content_img = Image.open(self.curr_content_file)
        prev_content_file = Image.open(self.prev_content_file)

        F_t = self.transform(prev_content_file).unsqueeze(0).to(self.device)
        F_tp1 = self.transform(curr_content_img).unsqueeze(0).to(self.device)
        S_t = self.transform(prev_stylized_frame).unsqueeze(0).to(self.device) # t
        S_tp1 = self.transform(stylized_frame).unsqueeze(0).to(self.device) # t+1

        with torch.no_grad():
            temporal_loss = self.temp_loss_fn(F_t, F_tp1, S_t, S_tp1)
            return temporal_loss.detach().cpu().item()  # Convert to float

    def _compute_content_loss(self, stylized_frame):
        stylized_frame = self.transform(stylized_frame)
        # loss_fn_alex = LPIPS(net='alex') # best forward scores
        # loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

        d = self.loss_fn_alex(stylized_frame.unsqueeze(0), self.curr_content_frame.unsqueeze(0))  # Add batch dimension
        return d.item()  # Convert tensor to Python float
    
    def _compute_style_loss(self, stylized_frame):
        def normalize_batch(batch):
            mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
            std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
            return (batch - mean) / std
        
        style_img = Image.open(self.style_file)
        style_tensor = normalize_batch(self.transform(style_img).unsqueeze(0))
        stylized_output_tensor = normalize_batch(self.transform(stylized_frame).unsqueeze(0))
        with torch.no_grad():
            style_features = self.vgg_model(style_tensor)
            style_grams = [gram_matrix(f) for f in style_features]
        g_s = style_grams[0]
        frame_features = self.vgg_model(stylized_output_tensor)
        g_f = gram_matrix(frame_features[0])
        loss_fn = nn.MSELoss()
        style_loss = loss_fn(g_f, g_s)
        return style_loss.item()
    
    def _compute_losses(self, stylized_frame, prev_stylized_frame):
        content_loss = self._compute_content_loss(stylized_frame)
        style_loss = self._compute_style_loss(stylized_frame)
        temporal_loss = self._compute_temporal_loss(stylized_frame, prev_stylized_frame)
        total_loss = self.content_weight *content_loss + self.style_weight * style_loss + self.temporal_weight * temporal_loss
        print(f"unweighted content_loss: {content_loss}, style_loss: {style_loss}, temporal_loss: {temporal_loss}\n")
        print(f"weighted content_loss: {self.content_weight * content_loss}, style_loss: {self.style_weight * style_loss}, temporal_loss: {self.temporal_weight * temporal_loss}\n")
        return content_loss, style_loss, temporal_loss, total_loss

    def step(self, delta_z, prev_modified_stylized_frame, prev_ori_stylized_frame, video_num):
        # Convert delta_z to float16 for consistency with PNP pipeline
        delta_z = delta_z.to(dtype=torch.float16)
        
        # Add delta_z to the latent at t = T
        modified_latent = self.curr_content_latents[-1, :, :, :].to(dtype=torch.float16) + delta_z
        content_latents = self.curr_content_latents.clone().to(dtype=torch.float16)
        content_latents[-1, :, :, :] = modified_latent
        modified_stylized_frame = self.pnp.run_pnp(content_latents, 
                                        self.style_latents.to(dtype=torch.float16), 
                                        self.style_file, 
                                        video_num,
                                        mode=self.mode,
                                        content_fn=self.output_modified_fn, 
                                        style_fn=self.style_file)[0]
        
        # Run PNP for original frame
        ori_stylized_frame = self.pnp.run_pnp(self.curr_content_latents.to(dtype=torch.float16), 
                                        self.style_latents.to(dtype=torch.float16),
                                        self.style_file,
                                        video_num,
                                        content_fn=self.output_ori_fn,
                                        style_fn=self.style_file)[0]

        if isinstance(modified_stylized_frame, torch.Tensor):
            modified_stylized_frame = modified_stylized_frame.float().clamp(0, 1)
            modified_stylized_frame = transforms.ToPILImage()(modified_stylized_frame)
        
        if isinstance(ori_stylized_frame, torch.Tensor):
            ori_stylized_frame = ori_stylized_frame.float().clamp(0, 1)
            ori_stylized_frame = transforms.ToPILImage()(ori_stylized_frame)

        # compute reward
        content, style, temporal, loss_modified = self._compute_losses(modified_stylized_frame, prev_modified_stylized_frame)
        content_ori, style_ori, temporal_ori, loss_ori = self._compute_losses(ori_stylized_frame, prev_ori_stylized_frame)
        reward = loss_ori - loss_modified

        return reward, content, style, temporal, loss_modified, content_ori, style_ori, temporal_ori, loss_ori, modified_stylized_frame, ori_stylized_frame