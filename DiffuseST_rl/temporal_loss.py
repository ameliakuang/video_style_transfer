import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large, raft_small

device = "cuda" if torch.cuda.is_available() else "cpu"

def warp(x, flow, padding_mode='border'):
    B, C, H, W = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device),
        torch.arange(W, device=x.device),
        indexing='ij'
    )
    grid = torch.stack((xx, yy), dim=0).float() # 2, H, W
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1) # B, 2, H, W

    vgrid = grid - flow
    # normalize to [-1, 1]
    vgrid[:, 0] = 2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0
    vgrid[:, 1] = 2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1) # B, H, W, 2

    return F.grid_sample(x, vgrid,
                         mode='bilinear',
                         padding_mode=padding_mode,
                         align_corners=True)


class TemporalConsistencyLossRAFT(nn.Module):
    """
    Bidirectional temporal consistency loss using RAFT optical flow.
    """
    def __init__(self,
                 small: bool = False,
                 loss_type: str = 'l1',
                 occlusion: bool = True,
                 occ_thresh_px: float = 1.0,
                 device: str = "cuda"):
        super().__init__()

        if small:
            self.flow_net = raft_small(pretrained=True, progress=False).to(device)
        else:
            self.flow_net = raft_large(pretrained=True, progress=False).to(device)

        self.flow_net.eval()

        if loss_type not in ('l1', 'l2'):
            raise ValueError('loss_type must be "l1" or "l2"')
        self.loss_type = loss_type
        self.occlusion = occlusion
        self.occ_thresh = occ_thresh_px
        self.device = device

        self.preprocess = T.Compose([
            T.ConvertImageDtype(torch.float32), # convert from [0,255] to [0,1]
            T.Normalize(0.5, 0.5), # map from [0,1] to [‑1,1]
        ])

    @torch.no_grad()
    def _compute_flow(self, im1, im2):
        B, C, H, W = im1.shape
        assert H % 8 == 0 and H % 8 == 0, "Image size must be divisible by 8 for RAFT"
        assert im1.shape == im2.shape, "Image batches shape should match"

        im1_p = self.preprocess(im1.clone()).to(self.device)
        im2_p = self.preprocess(im2.clone()).to(self.device)

        flow_list = self.flow_net(im1_p, im2_p)
        flow = flow_list[-1]

        return flow

    def _compute_loss(self, a, b):
        if self.loss_type == 'l1':
            return torch.abs(a - b)
        else:
            return (a - b) ** 2

    def forward(self, F_t, F_tp1, S_t, S_tp1):
        """
        F_t, F_tp1: content frames in [0,1], shape B×3×H×W
        S_t, S_tp1: stylized frames, shape B×3×H×W
        """
        with torch.no_grad():
            flow_fwd = self._compute_flow(F_t, F_tp1)
            flow_bwd = self._compute_flow(F_tp1, F_t)

            if self.occlusion:
                fb = warp(flow_bwd, flow_fwd)
                occ_mask = (torch.norm(flow_fwd + fb, dim=1, keepdim=True) < self.occ_thresh).float()
            else:
                occ_mask = 1.0

        # warp stylized images
        S_t_warp = warp(S_t, flow_fwd)
        S_tp1_warp = warp(S_tp1, flow_bwd)

        err_fwd = self._compute_loss(S_t_warp, S_tp1) * occ_mask
        err_bwd = self._compute_loss(S_tp1_warp, S_t) * occ_mask

        return 0.5 * (err_fwd.mean() + err_bwd.mean())
