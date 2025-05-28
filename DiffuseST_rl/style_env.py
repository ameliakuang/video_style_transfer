
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from lpips import LPIPS
from temporal_loss import TemporalConsistencyLossRAFT

class StyleEnv:
    def __init__(self, pnp, 
                 scheduler, 
                 device,
                 curr_content_latents, #[T, C, H, W]
                 style_latents,
                 curr_content_file,
                 style_file,
                 next_content_latents,
                 next_content_file,
                 content_weight=0.5,
                 temporal_weight=1.0):
        self.pnp = pnp
        self.scheduler = scheduler
        self.device = device

        self.curr_content_latents = curr_content_latents
        self.style_latents = style_latents
        self.curr_content_file = curr_content_file
        self.style_file = style_file
        self.next_content_latents = next_content_latents
        self.next_content_file = next_content_file

        # obtain output file names for debugging purposes - can be removed later
        self.output_modified_fn = f'{self.curr_content_file}_modified'
        self.output_ori_fn = f'{self.curr_content_file}_ori'
        
        self.content_weight = content_weight
        self.temporal_weight = temporal_weight

        self.transform = transforms.Compose([
            transforms.Resize((520, 520)),
            transforms.ToTensor(),  # scales to [0, 1]
        ])
        self.curr_content_frame = self.transform(Image.open(self.curr_content_file))
        # TODO: add content and style loss
    
    def _compute_temporal_loss(self, stylized_frame):
        pass

    def _compute_content_loss(self, stylized_frame):
        loss_fn_alex = LPIPS(net='alex') # best forward scores
        # loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

        d = loss_fn_alex(stylized_frame.unsqueeze(0), self.curr_content_frame.unsqueeze(0))  # Add batch dimension
        return d.item()  # Convert tensor to Python float
    
    def _compute_style_loss(self, stylized_frame):
        pass

    def step(self, delta_z):
        """
        Apply delta_z to the content latents to obtain modified latents
        Run PNP to obtain modified frame and original stylized frame
        Compute reward as difference between loss of modified frame and loss of original stylized frame
        TODO: consider adding a weight for delta_z to control the strength of the modified noise
        """
        # Add delta_z to the latent at t = T
        modified_latents = self.curr_content_latents[-1, :, :, :] + delta_z
        modified_stylized_frame = self.pnp.run_pnp(modified_latents, 
                                        self.style_latents, 
                                        self.style_file, 
                                        content_fn=self.output_modified_fn, 
                                        style_fn=self.style_file)[0]
        
        ori_stylized_frame = self.pnp.run_pnp(self.curr_content_latents, 
                                        self.style_latents,
                                        self.style_file,
                                        content_fn=self.output_ori_fn,
                                        style_fn=self.style_file)[0]
        # reward = compute_losses(modified_stylized_frame)
        # reward_ori = compute_losses(ori_stylized_frame)
        # reward = reward_modified - reward_ori
        return 0



        


        