import math
import torch
import random

import numpy as np
import SimpleITK as sitk

from skimage import exposure
from skimage import transform

from itertools import product


def randomized_pairs(
    ids: list[str],
    num_moving_imgs: int,
    num_fixed_imgs: int
) -> list[tuple]:
    """Receives the ids of all images, along with the number
    of images to be considered as the moving and fixed images.
    The method then randomly samples from the ids and pairs up
    the ids of the moving and fixed images
    
    :param ids:
        A list of strings containing the ids of all images;
        e.g., 0001, 0034, 0411

    :param num_moving_imgs:
        The number of images to be considered as the moving
        image

    :param num_fixed_imgs:
        The number of images to be considered as the atlas
        or fixed images
        
    :returns:
        A list of the form [(f_id1, m_id1), (f_id2, m_id2), ...],
        where m_id is the id of the moving image and the f_id is
        the id of the fixed image
    """
    # randomly choose the moving images
    moving_imgs_ids = random.sample(ids, num_moving_imgs)
    remained_ids = [i for i in ids if i not in moving_imgs_ids]
    fixed_ids = random.sample(remained_ids, num_fixed_imgs)
    
    fixed_moving_ids = list(product(fixed_ids, moving_imgs_ids))
    
    return fixed_moving_ids

def get_id_grid(shape: tuple) -> torch.Tensor:
    """Constructs a 2D or 3D identity grid

    :param shape:
        Spatial dimensions of the image.
        [D, H, W] or [H, W]

    :returns:
        An identity grid with shape [D, H, W, 3].
    """
    if len(shape) == 3:
        d, h, w = shape
        z, y, x = np.meshgrid(np.arange(d),
                              np.arange(h),
                              np.arange(w),
                              indexing='ij')
        id_grid = np.stack([x, y, z], 3)
    else:
        h, w = shape
        y, x = np.meshgrid(np.arange(h),
                           np.arange(w),
                              indexing='ij')
        id_grid = np.stack([x, y], 2)

    id_grid = torch.tensor(id_grid, dtype=torch.float32)
    
    return id_grid

def crop(
    imgs_list: list[np.ndarray],
    crop_x: int,
    crop_y: int,
    crop_z: int
) -> list[np.ndarray]:
    """Receives a list of images (fixed image, fixed segmentation,
    moving image, moving segmentation) and thecropping information
    and crops the images along each axis (if specified).
    
    :param imgs_list:
        A list containing the 3D images as numpy arrays.

    :param crop_x:
        The new size of the image along the x direction.

    :param crop_y:
        The new size of the image along the y direction.

    :param crop_z:
        The new size of the image along the z direction.
        
    :returns:
        A list of cropped images.
    """
    if crop_x is not None:
        imgs_list = crop_axis(imgs_list, crop_x, 0)
        
    if crop_y is not None:
        imgs_list = crop_axis(imgs_list, crop_y, 1)
        
    if crop_z is not None:
        imgs_list = crop_axis(imgs_list, crop_z, 2)
    
    return imgs_list
    
def crop_axis(
    imgs_list: list[np.ndarray],
    crop_size: int,
    axis: int=0
) -> list[np.ndarray]:
    """Crops the given list of image along the specified axis.
    
    :param imgs_list:
        A list of images to be cropped.

    :param crop_size:
        The new size of the image along the given axis.

    :param axis:
        Defaults to 0; The axis along which the cropping
        should be done (axis must be eiher 0, 1, or 2)

    :returns:
        A list of the cropped images along the specified axis
    """
    
    size = imgs_list[0].shape[axis]
    start = size // 2 - crop_size // 2
    start = start if start > 0 else 0
    
    end = start + crop_size
    end = end if end <= size else size
    if axis == 0:
        cropped_list = [img[start: end, :, :] for img in imgs_list]
    elif axis == 1:
        cropped_list = [img[:, start: end, :] for img in imgs_list]
    else:
        cropped_list = [img[:, :, start: end] for img in imgs_list]
    
    return cropped_list

def image_norm(img: np.ndarray) -> np.ndarray:
    """Applies min-max normalization on an image.

    :param img:
        The unnormalized input image

    :returns:
        The min-max intensity normalized image.
    """
    
    img = (img - img.min()) / (img.max() - img.min())
    
    return img

def resampler_by_transform(im_sitk, dvf_t, im_ref=None, 
                           default_pixel_value=0, 
                           interpolator=sitk.sitkBSpline):
    if im_ref is None:
        im_ref = sitk.Image(dvf_t.GetDisplacementField().GetSize(), sitk.sitkInt8)
        im_ref.SetOrigin(dvf_t.GetDisplacementField().GetOrigin())
        im_ref.SetSpacing(dvf_t.GetDisplacementField().GetSpacing())
        im_ref.SetDirection(dvf_t.GetDisplacementField().GetDirection())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(im_ref)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetTransform(dvf_t)
    out_im = resampler.Execute(im_sitk)
    return out_im

def resampler_sitk(image, spacing=None, scale=None, 
                   im_ref=None, im_ref_size=None, 
                   default_pixel_value=0, 
                   interpolator=sitk.sitkBSpline, 
                   dimension=3, offset=True):
    if offset:
        image = image + 1024

    image_sitk = sitk.GetImageFromArray(image)

    if spacing is None and scale is None:
        raise ValueError('spacing and scale cannot be both None')

    if spacing is None:
        spacing = tuple(i * scale for i in image_sitk.GetSpacing())
        if im_ref_size is None:
            im_ref_size = tuple(round(i / scale) for i in image_sitk.GetSize())

    elif scale is None:
        ratio = [spacing_dim / spacing[i] for i, spacing_dim in enumerate(image_sitk.GetSpacing())]
        if im_ref_size is None:
            # im_ref_size = tuple(math.ceil(size_dim * ratio[i]) - 1 for i, size_dim in enumerate(image_sitk.GetSize()))
            im_ref_size = tuple(math.ceil(size_dim * ratio[i]) for i, size_dim in enumerate(image_sitk.GetSize()))
    else:
        raise ValueError('spacing and scale cannot both have values')

    if im_ref is None:
        im_ref = sitk.Image(im_ref_size, sitk.sitkInt8)
        im_ref.SetOrigin(image_sitk.GetOrigin())
        im_ref.SetDirection(image_sitk.GetDirection())
        im_ref.SetSpacing(spacing)
    identity = sitk.Transform(dimension, sitk.sitkIdentity)
    resampled_sitk = resampler_by_transform(image_sitk, identity, im_ref=im_ref,
                                            default_pixel_value=default_pixel_value,
                                            interpolator=interpolator)
    
    resampled_img = sitk.GetArrayFromImage(resampled_sitk)

    return resampled_img

def affine_register(fixed_np, moving_np, moving_seg_np, spacing=(1.0,1.0,1.0), origin=(0,0,0)):
    # Convert numpy arrays to SimpleITK images
    fixed_img = sitk.GetImageFromArray(fixed_np.astype(np.float32))
    moving_img = sitk.GetImageFromArray(moving_np.astype(np.float32))
    moving_seg = sitk.GetImageFromArray(moving_seg_np.astype(np.int16))

    for im in [fixed_img, moving_img, moving_seg]:
        im.SetSpacing(spacing)
        im.SetOrigin(origin)

    # Registration method
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2)
    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsGradientDescent(learningRate=1.0,
                                               numberOfIterations=100,
                                               convergenceMinimumValue=1e-6,
                                               convergenceWindowSize=10)
    registration.SetOptimizerScalesFromPhysicalShift()

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img, moving_img, sitk.AffineTransform(fixed_img.GetDimension()),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)

    # Run registration
    final_transform = registration.Execute(fixed_img, moving_img)

    # Resample moving image
    registered_img = sitk.Resample(
        moving_img, fixed_img, final_transform,
        sitk.sitkLinear, 0.0, moving_img.GetPixelID()
    )
    # Resample segmentation (nearest neighbor)
    registered_seg = sitk.Resample(
        moving_seg, fixed_img, final_transform,
        sitk.sitkNearestNeighbor, 0, moving_seg.GetPixelID()
    )

    # Convert back to numpy
    registered_img_np = sitk.GetArrayFromImage(registered_img)
    registered_seg_np = sitk.GetArrayFromImage(registered_seg)

    return registered_img_np, registered_seg_np

def pad_and_resize(image, target_size=(256, 256), order=1):
    """
    Resizes image while maintaining aspect ratio via zero-padding using numpy.pad.
    """
    h, w = image.shape[:2]
    # Calculate scale to fit the longest side into target_size
    scale = min(target_size[0] / h, target_size[1] / w)
    
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized = transform.resize(image, (new_h, new_w), 
                               order=order, 
                               anti_aliasing=True if order > 0 else False, 
                               preserve_range=True)
    
    # Calculate padding to center the image
    pad_h = (target_size[0] - new_h) // 2
    pad_w = (target_size[1] - new_w) // 2
    
    # Padding widths for (top, bottom) and (left, right)
    # This handles the case where (target - new) is odd
    pad_width = ((pad_h, target_size[0] - new_h - pad_h),
                 (pad_w, target_size[1] - new_w - pad_w))
    
    # Use numpy.pad instead of skimage.util.pad
    if order == 0:
        resized = resized.astype(np.uint8)
    padded = np.pad(resized, pad_width=pad_width, mode='constant', constant_values=0)
    
    return padded

def camus_full_pipeline(image, target_size=(256, 256), n_iter=15, order=1):
    """
    Standardizes CAMUS images for registration models.
    Order: Resize -> SRAD -> CLAHE
    """
    # 1. Standardize Size
    img = pad_and_resize(image, target_size, order=order)
    if order == 0:
        return img
    
    # 2. SRAD (Speckle Reducing Anisotropic Diffusion)
    # Convert to float and normalize
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    dt = 0.15
    for _ in range(n_iter):
        # Using numpy.roll for neighborhood gradients
        n = np.roll(img, -1, axis=0) - img
        s = np.roll(img, 1, axis=0) - img
        e = np.roll(img, -1, axis=1) - img
        w = np.roll(img, 1, axis=1) - img
        
        # Conduction coefficient: reduces smoothing near high gradients (edges)
        # Higher kappa = more global smoothing
        kappa = 0.1 
        cN = np.exp(-(n/kappa)**2)
        cS = np.exp(-(s/kappa)**2)
        cE = np.exp(-(e/kappa)**2)
        cW = np.exp(-(w/kappa)**2)
        
        img += dt * (cN*n + cS*s + cE*e + cW*w)

    # 3. CLAHE (Contrast Enhancement)
    # equalize_adapthist expects 0-1 range and handles the enhancement
    img_final = exposure.equalize_adapthist(img, clip_limit=0.02)
    
    return img_final
