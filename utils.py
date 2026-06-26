import torch

import numpy as np
import torch.nn.functional as F


def warp(
    image: torch.Tensor,
    deformation_grid: torch.Tensor,
    nearest: bool=False
) -> torch.Tensor:
    """Implements image warping by a deformation grid.

    Args:
        image (torch.Tensor): 2D or 3D image with size [B, C, D, H, W] or [B, C, H, W]
    
        deformation_grid (torch.Tensor): 2D or 3D [-1, 1] normalized deformation grid
                                         with size [B, D, H, W, 3] or [B, H, W, 2]
    
        nearest (bool): Whether to use nearest neighbor interpolation.
                        Set to True for warping segmentation masks
    
    Returns:
        torch.Tensor: The warped image by the deformation grid
    """

    mode = 'nearest' if nearest else 'bilinear'

    warped = F.grid_sample(image, 
                           deformation_grid, 
                           padding_mode='reflection', 
                           align_corners=True, 
                           mode=mode)

    return warped


def jacobian_determinant(deformation_grid: torch.Tensor) -> torch.Tensor:
    """Computes the percentage of voxels/pixels having non-positive
    Jacobian determinant.
    
    Args:
        deformation_grid (torch.Tensor): 2D or 3D [-1, 1] normalized deformation grid
                                         with size [B, D, H, W, 3] or [B, H, W, 2]

    Returns:
        torch.Tensor: The percentage of negative determinants
    """
    if len(deformation_grid.size()) == 4:
        dy = deformation_grid[:, 1:, :-1, :] - deformation_grid[:, :-1, :-1, :]
        dx = deformation_grid[:, :-1, 1:, :] - deformation_grid[:, :-1, :-1, :]

        determinants = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    elif len(deformation_grid.size()) == 5:
        dy = deformation_grid[:, 1:, :-1, :-1, :] - deformation_grid[:, :-1, :-1, :-1, :]
        dx = deformation_grid[:, :-1, 1:, :-1, :] - deformation_grid[:, :-1, :-1, :-1, :]
        dz = deformation_grid[:, :-1, :-1, 1:, :] - deformation_grid[:, :-1, :-1, :-1, :]

        det0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
        det1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy [:, :, :,:, 2] * dz[:, :, :, :, 0])
        det2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy [:, :, :,:, 1] * dz[:, :, :, :, 0])

        determinants = det0 - det1 + det2
    
    num_neg_dets = len(determinants[determinants <= 0])
    total_points = torch.prod(torch.tensor(determinants.size(), device=determinants.device))
    
    neg_dets_percentage = num_neg_dets * 100 / total_points
    
    return neg_dets_percentage


def dsc_score(
    seg_map1: torch.Tensor,
    seg_map2: torch.Tensor,
    bg: bool=False,
    structured: bool=False
) -> torch.Tensor:
    """Receives the segmentation maps of two 2d or 3d images
    and computes the dice score coefficient between the two maps.

    Args:
        seg_map1 (torch.Tensor): [B, C, D, H, W] or [B, C, H, W]; first segmentation map

        seg_map2 (torch.Tensor): [B, C, D, H, W] or [B, C, H, W], second segmentation map

        bg (bool): Consideres background matches if set True. Default False

        structured (bool): If True, the dice over each structure is calculated,
                           otherwise the volumetric dice is computed. Defalut False

    Returns:
        torch.Tensor: The dice score between the two segmentation maps
    """
    if not structured:
        max_classes = max(torch.max(seg_map1), torch.max(seg_map2))
        
        batch_size = seg_map1.size(0)
        
        denominator = 2 * torch.prod(torch.tensor(seg_map1.shape[1:]))
        denominator = denominator.unsqueeze(0).to(seg_map1.device)
        if batch_size > 1:
            denominator = denominator.repeat(batch_size, 1)
        
        if not bg:
            denom_list = [seg_map1[i][seg_map1[i] != 0].size(0) + seg_map2[i][seg_map2[i] != 0].size(0) for i in range(batch_size)]
            denominator = torch.tensor(denom_list, device=seg_map1.device)
            seg_map1[seg_map1 == 0] = max_classes + 10
            seg_map2[seg_map2 == 0] = max_classes + 12
        
        numerator = 2 * (seg_map1 == seg_map2).flatten(start_dim=1).sum(dim=-1)
        
        dsc_score = (numerator / denominator).mean()
    else:
        dsc_score = 0.
        batch_size = seg_map1.size(0)
        
        for b in range(batch_size):
            labels = torch.unique(torch.cat((seg_map1[b], seg_map2[b])))
            labels = labels[torch.where(labels != 0)]

            dicem = torch.zeros(len(labels))
            for idx, lab in enumerate(labels):
                top = 2 * torch.sum(torch.logical_and(seg_map1 == lab, seg_map2[b] == lab))
                bottom = torch.sum(seg_map1 == lab) + torch.sum(seg_map2[b] == lab)
                bottom = torch.maximum(bottom, torch.tensor(torch.finfo(float).eps, device=seg_map1.device))
                dicem[idx] = top / bottom
            
            dsc_score += dicem.mean() / batch_size
    
    return dsc_score


def rolling_dice(
    network: torch.nn.Module,
    fixed_img: torch.Tensor,
    moving_img: torch.Tensor, 
    fixed_seg: torch.Tensor,
    moving_seg: torch.Tensor, 
    grid: torch.Tensor,
    save_path: str, 
    num_frames: int=100
) -> None:
    """Receives an instance of the SGDIR, the fixed and moving images,
    the fixed and moving segmentations masks, and computes the dice
    score along the trajectory of warping (i.e., the dice score of warped
    images at each time step). The results are saved into a
    fwd_rolling_dice.txt file

    Args:
        network (nn.Module): An instance of TPFM-DIR

        fixed_img (torch.Tensor): Fixed image with size [1, 1, D, H, W] or [1, 1, H, W]

        moving_img (torch.Tensor): Moving image with size [1, 1, D, H, W] or [1, 1, H, W]

        fixed_seg (torch.Tensor): Fixed image segmentation mask with size 
                                  [1, 1, D, H, W] or [1, 1, H, W]

        moving_seg (torch.Tensor): Moving image segmentation mask with size 
                                   [1, 1, D, H, W] or [1, 1, H, W]

        grid (torch.Tensor): The identty grid with size [1, D, H, W, 3] or [1, H, W, 2]

        save_path (str): The directory in which the results will be saved.

        num_frames (int): The number of time steps at which the dice score should be computed
                          Default 100.
    """  
    timesteps = torch.linspace(0, 1, num_frames)

    f_dice_scores = []
    f_jacs = []
    t_0 = torch.zeros(1, device=fixed_img.device)
    t_1 = torch.ones(1, device=fixed_img.device)
    
    for t in timesteps:
        t = t.view(1).to(fixed_seg.device)
        xyz = network(fixed_img, moving_img, grid, t_0, t)
        xyzr = network(fixed_img, moving_img, grid, t_1, t)
        
        seg_Jw = F.grid_sample(moving_seg,
                               xyz,
                               mode='nearest',
                               align_corners=True,
                               padding_mode='reflection')
        seg_Iw = F.grid_sample(fixed_seg,
                               xyzr,
                               mode='nearest',
                               align_corners=True,
                               padding_mode='reflection')
    
        f_dice = dsc_score(seg_Jw, seg_Iw, structured=False)
        f_jac = jacobian_determinant(xyz)
    
        f_dice_scores.append(f_dice.item())
        f_jacs.append(f_jac.item())
    
    np.savetxt(f'{save_path}/fwd_rolling_dice.txt', f_dice_scores)
    np.savetxt(f'{save_path}/fwd_rolling_jacs.txt', f_jacs)


def grid_denormalizer(deformation_grid: torch.Tensor) -> torch.Tensor:
    """De-normalizes a [-1, 1] normalized grid to its original spacing

    Args:
        deformation_grid (torch.Tensor): Normalized grid with size [B, D, H, W, 3] or [B, H, W, 2]

    Returns:
        torch.Tensor: Denormalized grid with size [B, D, H, W, 3] or [B, H, W, 2]
    """
    shape = deformation_grid.shape[1:-1][::-1]
    
    denormalized_grid = deformation_grid.clone()

    for i in range(len(shape)):
        denormalized_grid[..., i] = (denormalized_grid[..., i] * (shape[i] - 1)) / 2. + ((shape[i] - 1) / 2)
    
    return denormalized_grid
