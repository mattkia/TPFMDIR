"""Implementation of 2D/3D TPFM-DIR including the cocycle regularization
and time-dependent similarity measurements
"""

import torch
import random

import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from torch.utils.checkpoint import checkpoint


from metrics import (NCCLoss,
                     NGFLoss,
                     MINDLoss)


class TPFM(nn.Module):
    """Implementation of TPFM-DIR for 3D images
    """
    def __init__(
        self,
        backbone: nn.Module,
        loss_type: str='ncc'
    ) -> None:
        """
        Args:
            backbone (nn.Module): An arbitrary time-dependent architecture which receives
                                  an input tensor x (concatenation of fixed and moving images),
                                  an intial time step s, and an end time step t, and returns
                                  a displacement field with shape [B, 3, D, H, W].
            
            loss_type (str): The name of the loss function that should be used for training
                             the model; Supported: ncc, mse, l1, ngf, mind
        """
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
            fixed (torch.Tensor): Fixed image with size [B, 1, D, H, W]
            moving (torch.Tensor): Moving image with size [B, 1, D, H, W]
            id_grid (torch.Tensor): Identity grid with size [B, D, H, W, 3]
            s (torch.Tensor): Sampled initial time with size [B]
            t (torch.Tensor): Sampled end time with size [B]

        Returns:
            torch.Tensor: The deformation grid at time t with size [B, D, H, W, 3]
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
        """Computes the main flow component of the TPFM-DIR
        formulation

        Args:
            fixed (torch.Tensor): Fixed image with size [B, 1, D, H, W]
            moving (torch.Tensor): Moving image with size [B, 1, D, H, W]
            s (torch.Tensor): Sampled initial time with size [1]
            t (torch.Tensor): Sampled end time with size [1]

        Returns:
            torch.Tensor: The vector field at time t with size [B, 3, D, H, W]
        """
        u_in = torch.cat([fixed, moving], dim=1)

        disp = self.net(u_in, s, t)

        return disp

    def loss_flow(
        self,
        fixed: torch.Tensor,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        moving: torch.Tensor,
        id_grid: torch.Tensor,
        res: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the time-dependent image similarity loss and
        cocycle regularization

        Args:
            fixed (torch.Tensor): Fixed image with size [B, 1, D, H, W]
            moving (torch.Tensor): Moving image with size [B, 1, D, H, W]
            id_grid (torch.Tensor): Identity grid with size [B, D, H, W, 3]
            res (float): The resolution at which the loss is computed

        Returns:
            torch.Tensor, torch.Tensor: Similarity loss, cocycle loss
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples random time steps and computes:
        1. The warped moving and fixed images at that particular time
        2. The cocycle loss for the sampled time steps

        Args:
            fixed (torch.Tensor): Fixed image with size [B, 1, D, H, W]
            moving (torch.Tensor): Moving image with size [B, 1, D, H, W]
            id_grid (torch.Tensor): Identity grid with size [B, D, H, W, 3]

        Returns:
            torch.Tensor: Warped moving image with size [B, 1, D, H, W]
            torch.Tensor: Warped fixed image with size [B, 1, D, H, W]
            torch.Tensor: The cocycle loss
        """
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

            grid_0_s_t = id_grid + self.compose(flow_0_s, flow_s_t, id_grid).permute(0, 2, 3, 4, 1)
            
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

            grid_1_t_s = id_grid + self.compose(flow_1_t, flow_t_s, id_grid).permute(0, 2, 3, 4, 1)
            
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
        """Recieves a flow/displacement field and converts it into a
        deformation grid, ready to be used for warping an image.

        Args:
            flow (torch.Tensor): Flow/displacement field with size [B, 3, D, H, W]
            grid (torch.Tensor): The grid on which the flow should act with size
                                 [B, D, H, W, 3]

        Returns:
            torch.Tensor: The deformation grid with size [B, D, H, W, 3]
        """
        phi = grid + flow.permute(0, 2, 3, 4, 1)

        phi = self.grid_normalizer(phi)

        return phi

    def warp_with_grid(
        self,
        image: torch.Tensor,
        grid: torch.Tensor,
        normalize: bool=False
    ) -> torch.Tensor:
        """Warps an input image with the given normalized/unnormalized grid.
        If the grid is unnormalized the noromalize flag must be set True.

        Args:
            image (torch.Tensor): Input image with size [B, 1, D, H, W]
            grid (torch.Tensor): Deformation with size [B, D, H, W, 3]
            normalize (bool): If True, normalizes the grid into [-1, 1]
                              befor applying the grid
        
        Returns:
            torch.Tensor: The warped image with size [B, 1, D, H, W]
        """
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
        """Computes the composition of two flows: flow1 o flow2.

        Args:
            flow1 (torch.Tensor): Flow with size [B, 3, D, H, W]
            flow2 (torch.Tensor): Flow with size [B, 3, D, H, W]
            grid (torch.Tensor): Initial grid flow2 acts on, with size [B, D, H, W, 3]
        
        Returns:
            torch.Tensor: The composed flow with size [B, 3, D, H, W]
        """
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
        """Normalizes a deformation grid to [-1, 1]

        Args:
            grid (torch.Tensor): Unnormalized deformation with size [B, D, H, W, 3]
        
        Returns:
            torch.Tensor: Normalized deformation with size [B, D, H, W, 3]
        """
        _, d, h, w, _ = grid.size()

        grid[:, :, :, :, 0] = (grid[:, :, :, :, 0] - ((w - 1) / 2)) / (w - 1) * 2
        grid[:, :, :, :, 1] = (grid[:, :, :, :, 1] - ((h - 1) / 2)) / (h - 1) * 2
        grid[:, :, :, :, 2] = (grid[:, :, :, :, 2] - ((d - 1) / 2)) / (d - 1) * 2
        
        return grid

    def multistep_deform(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        grid: torch.Tensor,
        num_steps: int=8,
        forward: bool=True
    ) -> torch.Tensor:
        """Applies multistep composition of deformations used in the inference.

        Args:
            fixed (torch.Tensor): Fixed image with size [B, 1, D, H, W]
            moving (torch.Tensor): Moving image with size [B, 1, D, H, W]
            id_grid (torch.Tensor): Identity grid with size [B, D, H, W, 3]
            num_steps (int): Number of compositions, defaults to 8.
            forward (bool): If True, computes the moving-to-fixed deformation,
                            and computes fixed-to-moving if False
        
        Returns:
            torch.Tensor: The compositional deformation with size [B, D, H, W, 3]
        """
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
    """Implementation of TPFM-DIR for 2D images
    """
    def __init__(
        self, 
        backbone: nn.Module, 
        loss_type: str='ncc'
    ) -> None:
        """
        Args:
            backbone (nn.Module): An arbitrary time-dependent architecture which receives
                                  an input tensor x (concatenation of fixed and moving images),
                                  an intial time step s, and an end time step t, and returns
                                  a displacement field with shape [B, 2, H, W].
            
            loss_type (str): The name of the loss function that should be used for training
                             the model; Supported: ncc, mse, l1, ngf, mind
        """
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
            fixed (torch.Tensor): Fixed image with size [B, 1, H, W]
            moving (torch.Tensor): Moving image with size [B, 1, H, W]
            id_grid (torch.Tensor): Identity grid with size [B, H, W, 2]
            s (torch.Tensor): Sampled initial time with size [B]
            t (torch.Tensor): Sampled end time with size [B]

        Returns:
            torch.Tensor: The deformation grid at time t with size [B, H, W, 2]
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
        """Computes the main flow component of the TPFM-DIR
        formulation

        Args:
            fixed (torch.Tensor): Fixed image with size [B, 1, H, W]
            moving (torch.Tensor): Moving image with size [B, 1, H, W]
            s (torch.Tensor): Sampled initial time with size [1]
            t (torch.Tensor): Sampled end time with size [1]

        Returns:
            torch.Tensor: The vector field at time t with size [B, 2, H, W]
        """
        u_in = torch.cat([fixed, moving], dim=1)

        disp = self.net(u_in, s, t)

        return disp

    def loss_flow(
        self, 
        fixed: torch.Tensor,
        moving: torch.Tensor,
        id_grid: torch.Tensor,
        res: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the time-dependent image similarity loss and
        cocycle regularization

        Args:
            fixed (torch.Tensor): Fixed image with size [B, 1, H, W]
            moving (torch.Tensor): Moving image with size [B, 1, H, W]
            id_grid (torch.Tensor): Identity grid with size [B, H, W, 2]
            res (float): The resolution at which the loss is computed

        Returns:
            torch.Tensor, torch.Tensor: Similarity loss, cocycle loss
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
        """Samples random time steps and computes:
        1. The warped moving and fixed images at that particular time
        2. The cocycle loss for the sampled time steps

        Args:
            fixed (torch.Tensor): Fixed image with size [B, 1, H, W]
            moving (torch.Tensor): Moving image with size [B, 1, H, W]
            id_grid (torch.Tensor): Identity grid with size [B, H, W, 2]

        Returns:
            torch.Tensor: Warped moving image with size [B, 1, H, W]
            torch.Tensor: Warped fixed image with size [B, 1, H, W]
            torch.Tensor: The cocycle loss
        """
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
        """Recieves a flow/displacement field and converts it into a
        deformation grid, ready to be used for warping an image.

        Args:
            flow (torch.Tensor): Flow/displacement field with size [B, 2, H, W]
            grid (torch.Tensor): The grid on which the flow should act with size
                                 [B, H, W, 2]

        Returns:
            torch.Tensor: The deformation grid with size [B, H, W, 2]
        """
        phi = grid + flow.permute(0, 2, 3, 1)

        phi = self.grid_normalizer(phi)

        return phi
  
    def warp_with_grid(
        self,
        image: torch.Tensor,
        grid: torch.Tensor,
        normalize: bool=False
    ) -> torch.Tensor:
        """Warps an input image with the given normalized/unnormalized grid.
        If the grid is unnormalized the noromalize flag must be set True.

        Args:
            image (torch.Tensor): Input image with size [B, 1, H, W]
            grid (torch.Tensor): Deformation with size [B, H, W, 2]
            normalize (bool): If True, normalizes the grid into [-1, 1]
                              befor applying the grid
        
        Returns:
            torch.Tensor: The warped image with size [B, 1, H, W]
        """
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
        """Computes the composition of two flows: flow1 o flow2.

        Args:
            flow1 (torch.Tensor): Flow with size [B, 2, H, W]
            flow2 (torch.Tensor): Flow with size [B, 2, H, W]
            grid (torch.Tensor): Initial grid flow2 acts on, with size [B, H, W, 2]
        
        Returns:
            torch.Tensor: The composed flow with size [B, 2, H, W]
        """
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
        """Normalizes a deformation grid to [-1, 1]

        Args:
            grid (torch.Tensor): Unnormalized deformation with size [B, H, W, 2]
        
        Returns:
            torch.Tensor: Normalized deformation with size [B, H, W, 2]
        """
        _, h, w, _ = grid.size()

        grid[:, :, :, 0] = (grid[:, :, :, 0] - ((w - 1) / 2)) / (w - 1) * 2
        grid[:, :, :, 1] = (grid[:, :, :, 1] - ((h - 1) / 2)) / (h - 1) * 2
        
        return grid

    def multistep_deform(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        grid: torch.Tensor,
        num_steps: int=8,
        forward: bool=True
    ) -> torch.Tensor:
        """Applies multistep composition of deformations used in the inference.

        Args:
            fixed (torch.Tensor): Fixed image with size [B, 1, H, W]
            moving (torch.Tensor): Moving image with size [B, 1, H, W]
            id_grid (torch.Tensor): Identity grid with size [B, H, W, 2]
            num_steps (int): Number of compositions, defaults to 8.
            forward (bool): If True, computes the moving-to-fixed deformation,
                            and computes fixed-to-moving if False
        
        Returns:
            torch.Tensor: The compositional deformation with size [B, H, W, 2]
        """
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
