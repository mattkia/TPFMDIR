import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


def warp(
    image: torch.Tensor,
    deformation_grid: torch.Tensor,
    nearest: bool=False
) -> torch.Tensor:
    """Implements image warping by a deformation grid.

    :param image: 
        2D or 3D image
        Acceptable shapes: [B, C, D, H, W], [B, C, H, W]
    
    :param deformation_grid:
        2D or 3D [-1, 1] normalized deformation grid
        Acceptade shapes: [B, D, H, W, 3], [B, H, W, 2]
    
    :param nearest:
        Whether to use nearest neighbor interpolation.
        Set to True for warping segmentation masks
    
    :returns:
        The warped image by the deformation grid
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
    
    :param deformation_grid: 
        2D or 3D [-1, 1] normalized deformation grid
        Accepted shapes: [B, D, H, W, 3], [B, H, W, 2]

    :returns:
        The percentage of negative determinants
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

    :param seg_map1: 
        [B, C, D, H, W] or [B, C, H, W]; first segmentation map

    :param seg_map2:
        [B, C, D, H, W] or [B, C, H, W], second segmentation map

    :param bg:
        Consideres background matches if set True. Default False

    :param structured:
        If True, the dice over each structure is calculated,
        otherwise the volumetric dice is computed. Defalut False

    :returns:
        The dice score between the two segmentation maps
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

    :param network:
        An instanc of the SGDIR.

    :param fixed_img:
        Fixed image [1, 1, D, H, W] or [1, 1, H, W].

    :param moving_img:
        Moving image [1, 1, D, H, W] or [1, 1, H, W].

    :param fixed_seg:
        Fixed image segmentation mask [1, 1, D, H, W] or [1, 1, H, W].

    :param moving_seg:
        Moving image segmentation mask [1, 1, D, H, W] or [1, 1, H, W].

    :param grid:
        The identty grid [1, D, H, W, 3] or [1, H, W, 2].

    :param save_path:
        The directory in which the results will be saved.

    :param num_frames:
        The number of time steps at which the dice score should be computed
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


def evaluate_cocycle_property(
    network: nn.Module,
    fixed_img: torch.Tensor,
    moving_img: torch.Tensor,
    grid: torch.Tensor,
    save_path: str
) -> torch.Tensor:
    """
    This function receives the an instance of FlowNet3D along with the fixed and moving images and evaluates the percentage
    negative Jacobian determinants and how well the semigroup property holds. 
    The final results are saved into jacs.tx and errors.txt files which contain the information on Jacobian determinants
    and the semigroup property errors, respectively.
    
    Args:
        network (torch.nn.Module): an instanc of the FlowNet3D
        fixed_img (torch.Tensor): fixed image [1, 1, D, H, W]
        moving_img (torch.Tensor): moving image [1, 1, D, H, W]
        grid (torch.Tensor): the initial grid [1, D, H, W, 3]
        save_path (str): the directory in which the results will be saved
    """
    num_iters = 100
    s_vals = []
    t_vals = []
    error_means = []
    error_stds = []
    for _ in tqdm(range(num_iters)):
        s = torch.rand(1, device=fixed_img.device)
        t = torch.rand(1, device=fixed_img.device)

        if t < s:
            s, t = t, s
        
        s_vals.append(s.item())
        t_vals.append(t.item())
        cocycle_error = []
        for _ in range(num_iters):
            r = s + torch.rand(1, device=fixed_img.device) * (t - s)

            flow_s_t = (t - s) * network.flow_core(fixed_img, moving_img, s, t)
            flow_s_r = (r - s) * network.flow_core(fixed_img, moving_img, s, r)
            flow_r_t = (t - r) * network.flow_core(fixed_img, moving_img, r, t)

            phi_s_t = grid + flow_s_t.permute(0, 2, 3, 4, 1)

            composed_flow = network.compose(flow_s_r, flow_r_t, grid)
            composed_phi = grid + composed_flow.permute(0, 2, 3, 4, 1)

            error = torch.mean((phi_s_t - composed_phi) ** 2)
            cocycle_error.append(error.item())
        
        error_means.append(np.mean(cocycle_error))
        error_stds.append(np.std(cocycle_error))
    
    values = np.stack([s_vals, t_vals, error_means, error_stds], axis=0)
    np.savetxt(save_path, values)


def grid_denormalizer(deformation_grid: torch.Tensor) -> torch.Tensor:
    shape = deformation_grid.shape[1:-1][::-1]
    
    denormalized_grid = deformation_grid.clone()

    for i in range(len(shape)):
        denormalized_grid[..., i] = (denormalized_grid[..., i] * (shape[i] - 1)) / 2. + ((shape[i] - 1) / 2)
    
    return denormalized_grid
