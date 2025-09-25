# anomalib-nifti
Make anomalib compatible to .nii extension.


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
    # nifti_image = nib.load(file_path).get_fdata()
    # nifti_image_normalized = (nifti_image - np.min(nifti_image)) / (np.max(nifti_image) - np.min(nifti_image))
    # image = np.stack([nifti_image_normalized, nifti_image_normalized, nifti_image_normalized], axis=-1)  # (H, W, 3)
    # image = (255 * image).astype(np.uint8)
    # return Image.fromarray(image).convert("RGB")
    nifti_image = nib.load(file_path).get_fdata()
    nifti_image_normalized = 255 * (nifti_image - np.min(nifti_image)) / (np.max(nifti_image) - np.min(nifti_image))
    gray = nifti_image_normalized.astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_bone = Image.fromarray(rgb, mode='RGB').convert("RGB")
    return img_bone
```
**You can change the way you visualize 1-channel nifti image here**
 I used bone colormap here.


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
```

## 4. Change the way you convert 1-channel nifti file to 3-channel image
in anomalib_dataset.py line 30-40, currently using bone map, same as the previous png dataset.

## 5. Example usage (library report)
```python
from anomalib.models import Cfa
from anomalib.engine import Engine

from utils.folder import Folder
from lightning.pytorch.callbacks import EarlyStopping
from anomalib.metrics import Evaluator, AUROC

data_folder = "/home/user/lyeyang/projects/AnomalyOOD/BMAD/dataset/synthrad_v5"

cfg_model = {
    'backbone': 'resnet18',
}

datamodule = Folder(
    name="brain_nifti_resnet18_bone_image_auroc_nnn3_nhef3",
    root=data_folder,
    normal_dir="train/good",
    normal_test_dir="valid/good/img",
    abnormal_dir="valid/Ungood/img",
    mask_dir="valid/Ungood/label",
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=8,
    test_split_mode="from_dir",
    test_split_ratio=0,
    val_split_mode="same_as_test",
    val_split_ratio=1,
    seed=42,
    extensions=(".nii")
)

dm_test = Folder(
    name="brain_nifti_resnet18_bone_test",
    root=data_folder,
    normal_dir="train/good",
    normal_test_dir="test/good/img",
    abnormal_dir="test/Ungood/img",
    mask_dir="test/Ungood/label",
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=8,
    test_split_mode="from_dir",
    test_split_ratio=0,
    val_split_mode="same_as_test",
    val_split_ratio=1,
    seed=42,
    extensions=(".nii"),
)
early_stop_callback = EarlyStopping(
    monitor="image_AUROC",
    patience=10,
    mode="max"
)
evaluator = Cfa.configure_evaluator()

image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
evaluator.val_metrics.add_module("image_AUROC", image_auroc)
# evaluator.val_metrics.add_module("pixel_AUROC", pixel_auroc)

pre = Cfa.configure_pre_processor(image_size=224)

model = Cfa(**cfg_model, pre_processor=pre, post_processor=True,
                evaluator=evaluator
            )

engine = Engine(
    max_epochs=100,
    devices=[5],
    accelerator="gpu",
    callbacks=[early_stop_callback]
)

engine.fit(model=model, datamodule=datamodule)

valid_metrics = engine.test(model=model, datamodule=datamodule)
print(valid_metrics)

test_metrics = engine.test(model=model, datamodule=dm_test,)
print(test_metrics)
```
