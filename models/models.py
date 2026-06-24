import torch
import random

import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from torch.utils.checkpoint import checkpoint


from metrics import NCCLoss
from metrics import NGFLoss
from metrics import MINDLoss


class TPFM(nn.Module):
    """
    Implementation of time-dependent flow map based on two parameter cocycle property
    """
    def __init__(
        self,
        backbone: nn.Module,
        loss_type: str='ncc'
    ) -> None:
        super().__init__()

        self.loss_type = loss_type
        if loss_type == 'ncc':
            self.loss_fn = NCCLoss(win=9)
        elif loss_type == 'ngf':
            self.loss_fn = NGFLoss()
        elif loss_type == 'mind':
            self.loss_fn = MINDLoss(radius=5)
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        else:
            raise Exception('Invalid loss function')

        self.flow_based = True
        self.net = backbone

    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        id_grid: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, D, H, W]
            J (torch.Tensor): moving image with size [B, 1, D, H, W]
            xyz (torch.Tensor): identity grid with size [B, D, H, W, 3]
            t (torch.Tensor): sampled time with size [B]
        Returns:
            torch.Tensor: the deformation grid at time t with size [B, D, H, W, 3]
        """
        flow = (t - s) * self.flow_core(fixed, moving, s, t)

        phi_t = self.make_grid(flow, id_grid.clone())

        return phi_t
  
    def flow_core(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, D, H, W]
            J (torch.Tensor): moving image with size [B, 1, D, H, W]
            t (torch.Tensor): sampled time with size [1]
        Returns:
            torch.Tensor: the vector field at time t with size [B, 3, D, H, W]
        """
        u_in = torch.cat([fixed, moving], dim=1)

        velocity = self.net(u_in, s, t)

        return velocity

    def loss_flow(
        self,
        fixed: torch.Tensor,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        moving: torch.Tensor,
        id_grid: torch.Tensor,
        res: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, D, H, W]
            J (torch.Tensor): moving image with size [B, 1, D, H, W]
            id_grid (torch.Tensor): identity grid with size [B, D, H, W, 3]
            res (float): the resolution at which the ncc loss is computed
        Returns:
            torch.Tensor, torch.Tensor: ncc loss, semigroup loss
        """
        mw, fw, flow_loss = self.cocycle_loss(fixed, moving, id_grid)

        if res != 1:
            fw = F.interpolate(fw, scale_factor=res, mode='trilinear')
            mw = F.interpolate(mw, scale_factor=res, mode='trilinear')

        image_loss = res * self.loss_fn(mw, fw)

        return image_loss, flow_loss

    def cocycle_loss(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        id_grid: torch.Tensor
    ) -> torch.Tensor:
        t = torch.rand(1, device=fixed.device)
        s = torch.rand(1, device=fixed.device)
        t_0 = torch.zeros(1, device=fixed.device)
        t_1 = torch.ones(1, device=fixed.device)

        if s.item() > t.item():
            s, t = t, s

        if random.random() > 0.5:
            flow_0_s = s * self.flow_core(fixed, moving, t_0, s)
            grid_0_s = id_grid + flow_0_s.permute(0, 2, 3, 4, 1)

            flow_s_t = (t - s) * checkpoint(self.flow_core, fixed, moving, s, t, use_reentrant=False)

            if self.flow_based:
                grid_0_s_t = id_grid + self.compose(flow_0_s, flow_s_t, id_grid).permute(0, 2, 3, 4, 1)
            else:
                warping_grid = grid_0_s + flow_s_t.permute(0, 2, 3, 4, 1)
                grid_0_s_t = self.warp_with_grid(grid_0_s.permute(0, 4, 1, 2, 3),
                                                 warping_grid,
                                                 normalize=True).permute(0, 2, 3, 4, 1)
            
            flow_0_t = t * self.flow_core(fixed, moving, t_0, t)
            grid_0_t = id_grid + flow_0_t.permute(0, 2, 3, 4, 1)

            flow_1_t = (t - 1) * self.flow_core(fixed, moving, t_1, t)
            grid_1_t = id_grid + flow_1_t.permute(0, 2, 3, 4, 1)

            moving_warped = self.warp_with_grid(moving, grid_0_s_t, normalize=True)
            fixed_warped = self.warp_with_grid(fixed, grid_1_t, normalize=True)

            loss = torch.mean((grid_0_t - grid_0_s_t) ** 2)
        else:
            flow_1_t = (t - 1) * self.flow_core(fixed, moving, t_1, t)
            grid_1_t = id_grid + flow_1_t.permute(0, 2, 3, 4, 1)

            flow_t_s = (s - t) * checkpoint(self.flow_core, fixed, moving, t, s, use_reentrant=False)

            if self.flow_based:
                grid_1_t_s = id_grid + self.compose(flow_1_t, flow_t_s, id_grid).permute(0, 2, 3, 4, 1)
            else:
                warping_grid = grid_1_t + flow_t_s.permute(0, 2, 3, 4, 1)
                grid_1_t_s = self.warp_with_grid(grid_1_t.permute(0, 4, 1, 2, 3),
                                                 warping_grid,
                                                 normalize=True).permute(0, 2, 3, 4, 1)
            
            flow_1_s = (s - 1) * self.flow_core(fixed, moving, t_1, s)
            grid_1_s = id_grid + flow_1_s.permute(0, 2, 3, 4, 1)

            flow_0_s = s * self.flow_core(fixed, moving, t_0, s)
            grid_0_s = id_grid + flow_0_s.permute(0, 2, 3, 4, 1)

            moving_warped = self.warp_with_grid(moving, grid_0_s, normalize=True)
            fixed_warped = self.warp_with_grid(fixed, grid_1_t_s, normalize=True)

            loss = torch.mean((grid_1_s - grid_1_t_s) ** 2)
        
        return moving_warped, fixed_warped, loss

    def make_grid(
        self,
        flow: torch.Tensor,
        grid: torch.Tensor
    ) -> torch.Tensor:
        phi = grid + flow.permute(0, 2, 3, 4, 1)

        phi = self.grid_normalizer(phi)

        return phi

    def warp_with_grid(
        self,
        image: torch.Tensor,
        grid: torch.Tensor,
        normalize: bool=False
    ) -> torch.Tensor:
        if normalize:
            grid = self.grid_normalizer(grid.clone())
        
        warped = F.grid_sample(image,
                               grid,
                               padding_mode='border',
                               align_corners=True)
        
        return warped

    def compose(
        self,
        flow1: torch.Tensor,
        flow2: torch.Tensor,
        grid: torch.Tensor
    ):
        grid = grid + flow2.permute(0, 2, 3, 4, 1)

        grid = self.grid_normalizer(grid)

        composed_flow = F.grid_sample(flow1,
                                      grid,
                                      padding_mode='border',
                                      align_corners=True) + flow2

        return composed_flow
  
    def grid_normalizer(
        self,
        grid: torch.Tensor
    ) -> torch.Tensor:
        _, d, h, w, _ = grid.size()

        grid[:, :, :, :, 0] = (grid[:, :, :, :, 0] - ((w - 1) / 2)) / (w - 1) * 2
        grid[:, :, :, :, 1] = (grid[:, :, :, :, 1] - ((h - 1) / 2)) / (h - 1) * 2
        grid[:, :, :, :, 2] = (grid[:, :, :, :, 2] - ((d - 1) / 2)) / (d - 1) * 2
        
        return grid

    def integrate(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        grid: torch.Tensor,
        num_steps: int=8,
        forward: bool=True
    ) -> torch.Tensor:
        if forward:
            time_steps = torch.linspace(0, 1, num_steps + 1, device=grid.device)
        else:
            time_steps = torch.linspace(1, 0, num_steps + 1, device=grid.device)
        
        id_grid = grid.clone()
        t_start = time_steps[0].view(1)
        t_end = time_steps[1].view(1)
        flow = (t_end - t_start) * self.flow_core(fixed, moving, t_start, t_end)

        for i in range(1, num_steps):
            t_start = time_steps[i].view(1)
            t_end = time_steps[i + 1].view(1)
            rem_flow = (t_end - t_start) * self.flow_core(fixed, moving, t_start, t_end)

            flow = self.compose(flow, rem_flow, id_grid)

        grid = id_grid + flow.permute(0, 2, 3, 4, 1)
        grid = self.grid_normalizer(grid)

        return grid


class TPFM2D(nn.Module):
    """
    Implementation of both Time-Independet and Time-Dependent Phi network based on interpolative cmposition
    """
    def __init__(
        self, 
        backbone: nn.Module, 
        loss_type: str='ncc'
    ) -> None:
        super().__init__()

        self.loss_type = loss_type
        if loss_type == 'ncc':
            self.loss_fn = NCCLoss(win=9)
        elif loss_type == 'ngf':
            self.loss_fn = NGFLoss()
        elif loss_type == 'mind':
            self.loss_fn = MINDLoss(radius=5)
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        else:
            raise Exception('Invalid loss function')

        self.flow_based = True
        self.net = backbone
  
    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        id_grid: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, H, W]
            J (torch.Tensor): moving image with size [B, 1, H, W]
            grid (torch.Tensor): identity grid with size [B, H, W, 2]
            s (torch.Tensor): the initial sampled time with size [B]
            t (torch.Tensor): the final sampled time with size [B]
        Returns:
            torch.Tensor: the deformation grid at time t with size [B, H, W, 2]
        """
        flow = (t - s) * self.flow_core(fixed, moving, s, t)

        phi_t = self.make_grid(flow, id_grid.clone())

        return phi_t
  
    def flow_core(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            fixed (torch.Tensor): fixed image with size [B, 1, H, W]
            moving (torch.Tensor): moving image with size [B, 1, H, W]
            grid (torch.Tensor): the sampling grid with size [B, H, W, 2]
            s (torch.Tensor): the initial sampled time with size [B]
            t (torch.Tensor): the final sampled time with size [B]
        Returns:
            torch.Tensor: the vector field at time t with size [B, 2, H, W]
        """
        u_in = torch.cat([fixed, moving], dim=1)

        velocity = self.net(u_in, s, t)

        return velocity

    def loss_flow(
        self, 
        fixed: torch.Tensor,
        moving: torch.Tensor,
        id_grid: torch.Tensor,
        res: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, H, W]
            J (torch.Tensor): moving image with size [B, 1, H, W]
            id_grid (torch.Tensor): identity grid with size [B, H, W, 2]
            res (float): the resolution at which the ncc loss is computed
        Returns:
            torch.Tensor, torch.Tensor: ncc loss, semigroup loss
        """
        mw, fw, flow_loss = self.cocycle_loss(fixed, moving, id_grid)

        if res != 1:
            fw = F.interpolate(fw, scale_factor=res, mode='bilinear')
            mw = F.interpolate(mw, scale_factor=res, mode='bilinear')
        
        image_loss = res * self.loss_fn(mw, fw)

        return image_loss, flow_loss
  
    def cocycle_loss(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        id_grid: torch.Tensor
    ) -> torch.Tensor:
        t = torch.rand(1, device=fixed.device)
        s = torch.rand(1, device=fixed.device)
        t_0 = torch.zeros(1, device=fixed.device)
        t_1 = torch.ones(1, device=fixed.device)

        if s.item() > t.item():
            s, t = t, s

        if random.random() > 0.5:
            flow_0_s = s * self.flow_core(fixed, moving, t_0, s)
            grid_0_s = id_grid + flow_0_s.permute(0, 2, 3, 1)

            flow_s_t = (t - s) * checkpoint(self.flow_core, fixed, moving, s, t, use_reentrant=False)

            if self.flow_based:
                grid_0_s_t = id_grid + self.compose(flow_0_s, flow_s_t, id_grid).permute(0, 2, 3, 1)
            else:
                warping_grid = grid_0_s + flow_s_t.permute(0, 2, 3, 1)
                grid_0_s_t = self.warp_with_grid(grid_0_s.permute(0, 3, 1, 2),
                                                 warping_grid,
                                                 normalize=True).permute(0, 2, 3, 1)
            
            flow_0_t = t * self.flow_core(fixed, moving, t_0, t)
            grid_0_t = id_grid + flow_0_t.permute(0, 2, 3, 1)

            flow_1_t = (t - 1) * self.flow_core(fixed, moving, t_1, t)
            grid_1_t = id_grid + flow_1_t.permute(0, 2, 3, 1)

            moving_warped = self.warp_with_grid(moving, grid_0_s_t, normalize=True)
            fixed_warped = self.warp_with_grid(fixed, grid_1_t, normalize=True)

            loss = torch.mean((grid_0_t - grid_0_s_t) ** 2)
        else:
            flow_1_t = (t - 1) * self.flow_core(fixed, moving, t_1, t)
            grid_1_t = id_grid + flow_1_t.permute(0, 2, 3, 1)

            flow_t_s = (s - t) * checkpoint(self.flow_core, fixed, moving, t, s, use_reentrant=False)

            if self.flow_based:
                grid_1_t_s = id_grid + self.compose(flow_1_t, flow_t_s, id_grid).permute(0, 2, 3, 1)
            else:
                warping_grid = grid_1_t + flow_t_s.permute(0, 2, 3, 1)
                grid_1_t_s = self.warp_with_grid(grid_1_t.permute(0, 3, 1, 2),
                                                 warping_grid,
                                                 normalize=True).permute(0, 2, 3, 1)
            
            flow_1_s = (s - 1) * self.flow_core(fixed, moving, t_1, s)
            grid_1_s = id_grid + flow_1_s.permute(0, 2, 3, 1)

            flow_0_s = s * self.flow_core(fixed, moving, t_0, s)
            grid_0_s = id_grid + flow_0_s.permute(0, 2, 3, 1)

            moving_warped = self.warp_with_grid(moving, grid_0_s, normalize=True)
            fixed_warped = self.warp_with_grid(fixed, grid_1_t_s, normalize=True)

            loss = torch.mean((grid_1_s - grid_1_t_s) ** 2)
        
        return moving_warped, fixed_warped, loss

    def make_grid(
        self,
        flow: torch.Tensor,
        grid: torch.Tensor
    ) -> torch.Tensor:
        phi = grid + flow.permute(0, 2, 3, 1)

        phi = self.grid_normalizer(phi)

        return phi
  
    def warp_with_grid(
        self,
        image: torch.Tensor,
        grid: torch.Tensor,
        normalize: bool=False
    ) -> torch.Tensor:
        if normalize:
            grid = self.grid_normalizer(grid.clone())
        
        warped = F.grid_sample(image,
                               grid,
                               padding_mode='border',
                               align_corners=True)
        
        return warped

    def compose(
        self,
        flow1: torch.Tensor,
        flow2: torch.Tensor,
        grid: torch.Tensor
    ) -> torch.Tensor:
        grid = grid + flow2.permute(0, 2, 3, 1)

        grid = self.grid_normalizer(grid)

        composed_flow = F.grid_sample(flow1,
                                      grid,
                                      padding_mode='border',
                                      align_corners=True) + flow2

        return composed_flow
  
    def grid_normalizer(
        self,
        grid: torch.Tensor
    ) -> torch.Tensor:
        _, h, w, _ = grid.size()

        grid[:, :, :, 0] = (grid[:, :, :, 0] - ((w - 1) / 2)) / (w - 1) * 2
        grid[:, :, :, 1] = (grid[:, :, :, 1] - ((h - 1) / 2)) / (h - 1) * 2
        
        return grid

    def integrate(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        grid: torch.Tensor,
        num_steps: int=8,
        forward: bool=True
    ) -> torch.Tensor:
        if forward:
            time_steps = torch.linspace(0, 1, num_steps + 1, device=grid.device)
        else:
            time_steps = torch.linspace(1, 0, num_steps + 1, device=grid.device)
        
        id_grid = grid.clone()
        t_start = time_steps[0].view(1)
        t_end = time_steps[1].view(1)
        flow = (t_end - t_start) * self.flow_core(fixed, moving, t_start, t_end)

        for i in range(1, num_steps):
            t_start = time_steps[i].view(1)
            t_end = time_steps[i + 1].view(1)
            rem_flow = (t_end - t_start) * self.flow_core(fixed, moving, t_start, t_end)

            flow = self.compose(flow, rem_flow, id_grid)

        grid = id_grid + flow.permute(0, 2, 3, 1)
        grid = self.grid_normalizer(grid)

        return grid
