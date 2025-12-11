import os
import glob
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom

def center_crop_xy(img_np, mask_np, crop_h, crop_w):
    Z, H, W = img_np.shape
    cy, cx = H // 2, W // 2  # 图像中心点

    y1 = cy - crop_h // 2
    y2 = y1 + crop_h
    x1 = cx - crop_w // 2
    x2 = x1 + crop_w

    # 处理边界
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(H, y2)
    x2 = min(W, x2)

    img_np = img_np[:, y1:y2, x1:x2]
    mask_np = mask_np[:, y1:y2, x1:x2]

    return img_np, mask_np


def resize_3d(img_np, mask_np, target_size):
    """
    input: (Z,160,128)
    output: (128,128,128)
    """
    Z, H, W = img_np.shape
    td, th, tw = target_size
    zoom_factors = (td / Z, th / H, tw / W)

    img_np = zoom(img_np, zoom_factors, order=1)  # 线性插值
    mask_np = zoom(mask_np, zoom_factors, order=0) # mask 最近邻

    return img_np, mask_np

def apply_esophagus_window(img, w_center=40, w_width=400):
    lower = w_center - w_width // 2   # -160
    upper = w_center + w_width // 2   # 240
    img = np.clip(img, lower, upper)
    # 归一化到 0~1
    img = (img - lower) / (upper - lower)
    return img

class SAM3DTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, crop_h=160, crop_w=128, target_size=(128,128,128)):
        self.img_paths  = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))

        assert len(self.img_paths) == len(self.mask_paths)

        self.crop_h = crop_h
        self.crop_w = crop_w
        self.target_size = target_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_sitk  = sitk.ReadImage(self.img_paths[idx])
        mask_sitk = sitk.ReadImage(self.mask_paths[idx])

        img_np  = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        mask_np = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)

        original_shape = tuple(img_np.shape)

        # --- 食管癌纵隔窗 ---
        img_np = apply_esophagus_window(img_np)

        # 记录剪裁前mask体素
        mask_np_before_crop = mask_np.copy()
        orig_voxels = np.sum(mask_np_before_crop > 0)

        # 基于中心xy剪裁
        img_np, mask_np = center_crop_xy(img_np, mask_np, crop_h=self.crop_h, crop_w=self.crop_w)

        # 剪裁后mask体素
        after_crop_voxels = np.sum(mask_np > 0)
        if after_crop_voxels < orig_voxels:
            print(f"[Warning] Mask exceeds crop region for {self.img_paths[idx]}. "
                  f"Consider using larger crop_xy or using GT-based crop.")

        # resize
        img_np, mask_np = resize_3d(img_np, mask_np, target_size=self.target_size)

        # ---- 转成 Tensor ----
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()  # (1,D,H,W)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()  # (1,D,H,W)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "img_path": self.img_paths[idx],
            "original_shape": original_shape
        }
