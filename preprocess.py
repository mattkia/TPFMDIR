import math
import torch
import random

import numpy as np
import SimpleITK as sitk

from itertools import product
from skimage import transform


def randomized_pairs(
    ids: list[str],
    num_moving_imgs: int,
    num_fixed_imgs: int
) -> list[tuple]:
    """Receives the ids of all images, along with the number
    of images to be considered as the moving and fixed images.
    The method then randomly samples from the ids and pairs up
    the ids of the moving and fixed images
    
    Args:
        ids (list[str]): A list of strings containing the ids of all images;
                         e.g., 0001, 0034, 0411

        num_moving_imgs (int): The number of images to be considered as the moving image

        num_fixed_imgs (int): The number of images to be considered as the atlas or fixed images
        
    Returns:
        list[tuple[str, str]]: A list of the form [(f_id1, m_id1), (f_id2, m_id2), ...],
                               where m_id is the id of the moving image and the f_id is
                               the id of the fixed image
    """
    # Randomly choose the moving images
    moving_imgs_ids = random.sample(ids, num_moving_imgs)
    remained_ids = [i for i in ids if i not in moving_imgs_ids]
    fixed_ids = random.sample(remained_ids, num_fixed_imgs)
    
    fixed_moving_ids = list(product(fixed_ids, moving_imgs_ids))
    
    return fixed_moving_ids

def get_id_grid(shape: tuple) -> torch.Tensor:
    """Constructs a 2D or 3D identity grid

    Args:
        shape (int): The shape of the grid; [D, H, W] or [H, W]

    Returns:
        torch.Tensor: An identity grid with shape [D, H, W, 3] or [H, W, 2].
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
    
    Args:
        imgs_list (list[np.ndarray]): A list containing the 3D images as numpy arrays.
        crop_x (int): The new size of the image along the x direction.
        crop_y (int): The new size of the image along the y direction.
        crop_z (int): The new size of the image along the z direction.
        
    Returns:
        list[np.ndarray]: A list of cropped images.
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
    
    Args:
        imgs_list (list[np.ndarray]): A list of images to be cropped.

        crop_size (int): The new size of the image along the given axis.

        axis (int): Defaults to 0; The axis along which the cropping
              should be done (axis must be eiher 0, 1, or 2)

    Returns:
        list[np.ndarray]: A list of the cropped images along the specified axis
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

    Args:
        img (np.ndarray): The unnormalized input image

    Returns:
        np.ndarray: The min-max intensity normalized image.
    """
    
    img = (img - img.min()) / (img.max() - img.min())
    
    return img

def resampler_by_transform(
    im_sitk: sitk.Image,
    dvf_t: sitk.Transform,
    im_ref: sitk.Image=None, 
    default_pixel_value: int=0, 
    interpolator=sitk.sitkBSpline
) -> sitk.Image:
    """Resamples a SimpleITK image with a given transform.

    Args:
        im_sitk (sitk.Image): The image to resample.

        dvf_t: A SimpleITK transform object used for resampling.

        im_ref (sitk.Image | None): Optional reference image that defines
                                    the output size, spacing, origin, and
                                    direction. If None, the reference is created
                                    from the transform's displacement field.

        default_pixel_value (int): Pixel value used outside the image bounds.

        interpolator: The SimpleITK interpolator to use for resampling.

    Returns:
        sitk.Image: The resampled image.
    """
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

def resampler_sitk(
    image: np.ndarray,
    spacing: tuple[float]=None,
    scale: float=None, 
    im_ref: sitk.Image=None,
    im_ref_size: tuple[int]=None, 
    default_pixel_value: int=0, 
    interpolator=sitk.sitkBSpline, 
    dimension: int=3,
    offset: bool=True
) -> np.ndarray:
    """Resamples a NumPy image using SimpleITK with optional spacing and scale.

    Args:
        image: A NumPy array representing the image to resample.

        spacing (tuple[float] | None): Output image spacing. If provided, the
            image will be resampled to this spacing.

        scale (float | None): Scaling factor applied to the image spacing when
            spacing is not provided.

        im_ref (sitk.Image | None): Optional reference SimpleITK image defining
            the output geometry.

        im_ref_size (tuple[int] | None): Optional reference image size used
            when `im_ref` is not provided.

        default_pixel_value (int): Value used for pixels outside the image
            bounds during resampling.

        interpolator: SimpleITK interpolator for resampling.

        dimension (int): The image dimension for the transform.

        offset (bool): Whether to offset the image intensities by 1024 prior
            to resampling.

    Returns:
        np.ndarray: The resampled image as a NumPy array.
    """
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

def affine_register(
    fixed_np: np.ndarray,
    moving_np: np.ndarray,
    moving_seg_np: np.ndarray,
    spacing: list | tuple=(1.0, 1.0, 1.0),
    origin=(0, 0, 0)
) -> tuple[np.ndarray, np.ndarray]:
    """Performs affine registration of a moving volume to a fixed volume.

    Args:
        fixed_np (np.ndarray): A NumPy array representing the fixed image.
        moving_np (np.ndarray): A NumPy array representing the moving image to be registered.
        moving_seg_np (np.ndarray): A NumPy array containing the moving image segmentation.
        spacing (tuple[float]): Image spacing for all input volumes.
        origin (tuple[float]): Image origin for all input volumes.

    Returns:
        tuple[np.ndarray, np.ndarray]: The registered moving image and the
                                       registered segmentation as NumPy arrays.
    """
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

def pad_and_resize(
    image: np.ndarray,
    target_size: tuple[int]=(256, 256),
    order: int=1
) -> np.ndarray:
    """Resizes image while maintaining aspect ratio via zero-padding

    Args:
        image (np.ndarray): 2D Image to be resized
        target_size (tuple[int]): The target size of the image
        order (int): Interpolation type (1 for bilinear, 0 for nearest)
    
    Returns:
        np.ndarray: The resized image with the target size
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
    
    if order == 0:
        resized = resized.astype(np.uint8)
    padded = np.pad(resized, pad_width=pad_width, mode='constant', constant_values=0)
    
    return padded
