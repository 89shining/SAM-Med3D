# test_dataset_nocrop.py

import os
import glob
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset
import torchio as tio
from scipy.ndimage import zoom


# ------------------------------------------------
#   zoom 和训练方式一致（基于 subject['image'].data）
# ------------------------------------------------
class ResizeWithZoom:
    def __init__(self, target_size=(128,128,128)):
        self.target_z, self.target_h, self.target_w = target_size

    def __call__(self, subject):
        img = subject['image'].data[0].cpu().numpy()
        mask = subject['label'].data[0].cpu().numpy()

        z, h, w = img.shape
        zoom_factor = [
            self.target_z / z,
            self.target_h / h,
            self.target_w / w
        ]

        img_resized  = zoom(img, zoom_factor, order=3)
        mask_resized = zoom(mask, zoom_factor, order=0)

        subject['image'].set_data(torch.from_numpy(img_resized ).unsqueeze(0))
        subject['label'].set_data(torch.from_numpy(mask_resized).unsqueeze(0))
        return subject


# ------------------------------------------------
#   Test Dataset（完全对齐训练，不做 crop）
# ------------------------------------------------
class SAM3DTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths  = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))

        assert len(self.img_paths) == len(self.mask_paths)

        # 训练时的 transform 去掉随机增强
        self.transform = tio.Compose([
            tio.ToCanonical(),
            ResizeWithZoom(target_size=(128,128,128)),
            tio.ZNormalization(masking_method=lambda x: x > 0),
        ])

    def __len__(self):
        return len(self.img_paths)

    def get_3d_box_from_mask(self, mask):
        m = mask.squeeze(0).cpu().numpy()
        coords = np.argwhere(m > 0)
        if coords.shape[0] == 0:
            return torch.tensor([[0,0,0],[0,0,0]], dtype=torch.float32)

        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        return torch.tensor([[x_min,y_min,z_min],
                             [x_max,y_max,z_max]], dtype=torch.float32)

    def __getitem__(self, idx):
        img_sitk  = sitk.ReadImage(self.img_paths[idx])
        mask_sitk = sitk.ReadImage(self.mask_paths[idx])

        img_np  = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        mask_np = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)

        original_shape = tuple(img_np.shape)

        # ---- 构建 Subject ----
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.from_numpy(img_np ).unsqueeze(0)),
            label=tio.LabelMap (tensor=torch.from_numpy(mask_np).unsqueeze(0)),
        )

        # ---- transform（resize、normalize）----
        subject = self.transform(subject)

        img_tensor  = subject['image'].data.float()   # (1,128,128,128)
        mask_tensor = subject['label'].data.float()

        box = self.get_3d_box_from_mask(mask_tensor)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "box": box,
            "img_path": self.img_paths[idx],
            "original_shape": original_shape
        }
