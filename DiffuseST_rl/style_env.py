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
                 content_weight=1,
                 temporal_weight=10.0,
                 style_weight=0.1,
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
        

    
    def _compute_temporal_loss(self, 
                               stylized_frame,
                               prev_stylized_frame):
        if prev_stylized_frame is None:
            return 0.0
        temp_loss_fn = TemporalConsistencyLossRAFT(
            small=True,
            loss_type='l1',
            occlusion=True,
            occ_thresh_px=1.0,
            device=self.device)

        curr_content_img = Image.open(self.curr_content_file)
        prev_content_file = Image.open(self.prev_content_file)

        F_t = self.transform(prev_content_file).unsqueeze(0).to(self.device)
        F_tp1 = self.transform(curr_content_img).unsqueeze(0).to(self.device)
        S_t = self.transform(prev_stylized_frame).unsqueeze(0).to(self.device) # t
        S_tp1 = self.transform(stylized_frame).unsqueeze(0).to(self.device) # t+1

        with torch.no_grad():
            temporal_loss = temp_loss_fn(F_t, F_tp1, S_t, S_tp1)
            return temporal_loss.detach().cpu().item()  # Convert to float

    def _compute_content_loss(self, stylized_frame):
        stylized_frame = self.transform(stylized_frame)
        loss_fn_alex = LPIPS(net='alex') # best forward scores
        # loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

        d = loss_fn_alex(stylized_frame.unsqueeze(0), self.curr_content_frame.unsqueeze(0))  # Add batch dimension
        return d.item()  # Convert tensor to Python float
    
    def _compute_style_loss(self, stylized_frame):
        def normalize_batch(batch):
            mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
            std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
            return (batch - mean) / std
        
        style_img = Image.open(self.style_file)
        style_tensor = normalize_batch(self.transform(style_img).unsqueeze(0))
        stylized_output_tensor = normalize_batch(self.transform(stylized_frame).unsqueeze(0))
        feature_layers = [1, 6, 11, 20, 29]
        vgg = VGGFeatures(feature_layers)
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        with torch.no_grad():
            style_features = vgg(style_tensor)
            style_grams = [gram_matrix(f) for f in style_features]
        g_s = style_grams[0]
        frame_features = vgg(stylized_output_tensor)
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

    def step(self, delta_z, prev_modified_stylized_frame, prev_ori_stylized_frame):
        """
        Apply delta_z to the content latents to obtain modified latents
        Run PNP to obtain modified frame and original stylized frame
        Compute reward as difference between loss of modified frame and loss of original stylized frame
        TODO: consider adding a weight for delta_z to control the strength of the modified noise
        """
        # Add delta_z to the latent at t = T
        modified_latent = self.curr_content_latents[-1, :, :, :] + delta_z
        content_latents = self.curr_content_latents.clone()
        content_latents[-1, :, :, :] = modified_latent
        modified_stylized_frame = self.pnp.run_pnp(content_latents, 
                                        self.style_latents, 
                                        self.style_file, 
                                        content_fn=self.output_modified_fn, 
                                        style_fn=self.style_file)[0]
        
        ori_stylized_frame = self.pnp.run_pnp(self.curr_content_latents, 
                                        self.style_latents,
                                        self.style_file,
                                        content_fn=self.output_ori_fn,
                                        style_fn=self.style_file)[0]

        # compute reward
        content, style, temporal, loss_modified = self._compute_losses(modified_stylized_frame, prev_modified_stylized_frame)
        content_ori, style_ori, temporal_ori, loss_ori = self._compute_losses(ori_stylized_frame, prev_ori_stylized_frame)
        reward = loss_ori - loss_modified

        return reward, content, style, temporal, loss_modified, content_ori, style_ori, temporal_ori, loss_ori, modified_stylized_frame, ori_stylized_frame