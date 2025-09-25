
## 1. Use New folder dataset class to load NIfTI images
copy `anomalib_data_module.py`, `anomalib_dataset.py`, `folder_dataset.py`, `folder.py` to local and import `folder` from `folder.py` whenever needed.

## 2. Modify `item_visualizer.py` to read NIfTI images
File ~/miniconda3/envs/anomalib/lib/python3.10/site-packages/anomalib/visualization/image/item_visualizer.py:340, in visualize_image_item(item, fields, overlay_fields, field_size, fields_config, overlay_fields_config, text_config)

replace
```python
--> 340     image = Image.open(image_path).convert("RGB")
```

to
```python
--> 340         if image_path.endswith(".nii"):
                    image = read_nifti_image(image_path, as_tensor=False)
                else:
                    image = Image.open(image_path).convert("RGB")
```

Add to the end of `item_visualizer.py`:
```python
from torchvision.transforms.v2.functional import to_dtype, to_image
import torch
import nibabel as nib
import numpy as np
def read_nifti_image(file_path: str, as_tensor: bool = False) -> torch.Tensor | np.ndarray:
    nifti_image = nib.load(file_path).get_fdata()
    image = np.stack([nifti_image, nifti_image, nifti_image], axis=-1)  # (H, W, 3)
    return Image.fromarray((image).astype(np.uint8))
```

**You can change the way you deal with 1-channel nifti image here**

## 3. Modify `visualizer.py` to be compatible with nii extension
File ~/miniconda3/envs/anomalib/lib/python3.10/site-packages/anomalib/visualization/image/visualizer.py:223, in ImageVisualizer.on_test_batch_end(***failed resolving arguments***)

from
```python
                image.save(filename)
```

to 

```python
                if filename.suffix == ".nii":
                    filename = filename.with_suffix(".png")
                image.save(filename)
````