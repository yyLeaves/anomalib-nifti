# from __future__ import annotations

from anomalib.data import AnomalibDataset
from torchvision.tv_tensors import Mask
import torch
import nibabel as nib
import numpy as np
from pathlib import Path

from anomalib import TaskType
from anomalib.data.dataclasses import DatasetItem, ImageBatch, ImageItem
from anomalib.data.utils import LabelName
from torchvision.transforms.v2.functional import to_dtype, to_image
from anomalib.data.utils import LabelName, read_image, read_mask

def read_nifti_image(file_path: str, as_tensor: bool = False) -> torch.Tensor | np.ndarray:
    """Read a NIfTI image from the given file path.

    Args:
        file_path (str): Path to the NIfTI file.
        as_tensor (bool, optional): Whether to return the image as a torch.Tensor. Defaults to False.

    Returns:
        torch.Tensor | np.ndarray: The image read from the NIfTI file.
    """

    #image = Image.open(path).convert("RGB")
    #return to_dtype(to_image(image), torch.float32, scale=True) if as_tensor else np.array(image) / 255.0
    # (H, W, D)
    import cv2
    from PIL import Image
    nifti_image = nib.load(file_path).get_fdata()
    normalized_image = (nifti_image - np.min(nifti_image)) / (np.max(nifti_image) - np.min(nifti_image))
    image = np.stack([normalized_image, normalized_image, normalized_image], axis=-1)  # (H, W, 3)
    gray = (normalized_image * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # img_bone = Image.fromarray(rgb, mode='RGB').convert("RGB")
    img_bone_normalized = rgb / 255.0
    return to_dtype(to_image(img_bone_normalized), torch.float32, scale=True) if as_tensor else img_bone_normalized / 255.0


def read_nifti_mask(path: str | Path, as_tensor: bool = False) -> torch.Tensor | np.ndarray:
    nifti_mask = nib.load(path).get_fdata()
    return Mask(to_image(nifti_mask).squeeze() > 0, dtype=torch.uint8) if as_tensor else np.array(nifti_mask)


def __getitem__(self, index: int) -> DatasetItem:
    """Get dataset item for the given index.

    Args:
        index (int): Index to get the item.

    Returns:
        DatasetItem: Dataset item containing image and ground truth (if available).

    Example:
        >>> dataset = AnomalibDataset()
        >>> item = dataset[0]
        >>> isinstance(item.image, torch.Tensor)
        True
    """
    image_path = self.samples.iloc[index].image_path
    mask_path = self.samples.iloc[index].mask_path
    label_index = self.samples.iloc[index].label_index

    # Read the image
    image = read_nifti_image(image_path, as_tensor=True)

    # Initialize mask as None
    gt_mask = None

    # Process based on task type
    if self.task == TaskType.SEGMENTATION:
        if label_index == LabelName.NORMAL:
            # Create zero mask for normal samples
            gt_mask = Mask(torch.zeros(image.shape[-2:])).to(torch.uint8)
        elif label_index == LabelName.ABNORMAL:
            # Read mask for anomalous samples
            gt_mask = read_nifti_mask(mask_path, as_tensor=True)
        # For UNKNOWN, gt_mask remains None

    # Apply augmentations if available
    if self.augmentations:
        if self.task == TaskType.CLASSIFICATION:
            image = self.augmentations(image)
        elif self.task == TaskType.SEGMENTATION:
            # For augmentations that require both image and mask:
            # - Use a temporary zero mask for UNKNOWN samples
            # - But preserve the final gt_mask as None for UNKNOWN
            temp_mask = gt_mask if gt_mask is not None else Mask(torch.zeros(image.shape[-2:])).to(torch.uint8)
            image, augmented_mask = self.augmentations(image, temp_mask)
            # Only update gt_mask if it wasn't None before augmentations
            if gt_mask is not None:
                gt_mask = augmented_mask

    # Create gt_label tensor (None for UNKNOWN)
    gt_label = None if label_index == LabelName.UNKNOWN else torch.tensor(label_index)

    # Return the dataset item
    return ImageItem(
        image=image,
        gt_mask=gt_mask,
        gt_label=gt_label,
        image_path=image_path,
        mask_path=mask_path,
    )

# modify AnomalibDataset.__getitem__
AnomalibDataset.__getitem__ = __getitem__