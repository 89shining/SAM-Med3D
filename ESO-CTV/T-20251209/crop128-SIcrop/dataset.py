"""
image/CTV.nii.gz 512 → Crop XY (160×128) → GT box crop Z → Z padding → XY resize → (128×128×128)
"""

"""
基于GT的 框裁剪上下界
"""

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

def apply_esophagus_window(img, w_center=40, w_width=400):
    lower = w_center - w_width // 2   # -160
    upper = w_center + w_width // 2   # 240
    img = np.clip(img, lower, upper)
    # 归一化到 0~1
    img = (img - lower) / (upper - lower)
    return img


class SAM3DDataset(Dataset):
    def __init__(self, img_dir, mask_dir, crop_h=160, crop_w=128, target_size=(128,128,128)):
        """
        Args:
            img_dir (str): 图像文件夹路径 (imagesTr)
            mask_dir (str): 标签文件夹路径 (labelsTr)
        """
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
        assert len(self.img_paths) == len(self.mask_paths), "图像和mask数量不一致！"

        self.crop_h = crop_h
        self.crop_w = crop_w
        self.target_size = target_size

        # 预处理
        self.transform = tio.Compose([
            tio.RandomFlip(axes=(1, 2)),  # 数据增强：随机翻转xy轴
        ])

    def __len__(self):
        return len(self.img_paths)

    def get_gt_z_bounds(self, mask_np):
        """
        返回肿瘤在 Z 方向的上下界
        mask_np: (Z,H,W)
        """
        coords = np.argwhere(mask_np > 0)
        if coords.shape[0] == 0:
            raise ValueError("Mask is empty!")
        z_min = coords[:, 0].min()
        z_max = coords[:, 0].max()
        return z_min, z_max

    def __getitem__(self, idx):
        # 读取原始图像和标签
        img_sitk = sitk.ReadImage(self.img_paths[idx])
        mask_sitk = sitk.ReadImage(self.mask_paths[idx])

        img_np = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        mask_np = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)
        # print("原始图像 shape:", img_np.shape)  # (Z, H, W)
        # print("原始掩码 shape:", mask_np.shape)

        # --- 食管癌纵隔窗 ---
        img_np = apply_esophagus_window(img_np)

        # 记录剪裁前mask体素
        mask_np_before_crop = mask_np.copy()
        orig_voxels = np.sum(mask_np_before_crop > 0)

        # 获取 GT 上下界
        z_min, z_max = self.get_gt_z_bounds(mask_np)

        # 基于中心xy剪裁
        img_np, mask_np = center_crop_xy(img_np, mask_np, crop_h=self.crop_h, crop_w=self.crop_w)

        # 裁剪 Z
        img_np = img_np[z_min:z_max + 1]  # shape → (Z_gt,160,128)
        mask_np = mask_np[z_min:z_max + 1]

        # 剪裁后mask体素
        after_crop_voxels = np.sum(mask_np > 0)
        if after_crop_voxels < orig_voxels:
            print(f"[Warning] Mask exceeds crop region for {self.img_paths[idx]}. "
                  f"Consider using larger crop_xy or using GT-based crop.")

        Z_gt = img_np.shape[0]
        if Z_gt > 128:
            raise ValueError(f"GT Z exceeds 128 ({Z_gt}), cannot pad!")

        # Z pad 到 128（前后对称 padding）  （128, 160, 128)
        pad_before = (128 - Z_gt) // 2
        pad_after = 128 - Z_gt - pad_before

        img_np = np.pad(img_np, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
        mask_np = np.pad(mask_np, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')

        # resize
        img_np = zoom(img_np, (1, 128 / 160, 128 / 128), order=1)
        mask_np = zoom(mask_np, (1, 128 / 160, 128 / 128), order=0)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.from_numpy(img_np).unsqueeze(0)),
            label=tio.LabelMap(tensor=torch.from_numpy(mask_np).unsqueeze(0))
        )

        # print("Subject image shape:", subject['image'].data.shape)  # (1, Z, H, W)
        # print("Subject mask shape:", subject['label'].data.shape)

        #  应用 transform
        subject = self.transform(subject)

        img_tensor = subject['image'].data.float()  # (1,D,H,W)
        mask_tensor = subject['label'].data.float()  # (1,D,H,W)

        # print("Transform 后 image shape:", subject['image'].data.shape)
        # print("Transform 后 mask shape:", subject['label'].data.shape)

        return {
            "image": img_tensor,   # (1,D,H,W)
            "mask": mask_tensor,   # (1,D,H,W)
        }

# ================== 测试 ==================
if __name__ == "__main__":
    img_dir = r"C:\Users\dell\Desktop\dataset\train\imagesTr"
    mask_dir = r"C:\Users\dell\Desktop\dataset\train\labelsTr"

    dataset = SAM3DDataset(img_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        print("Image:", batch["image"].shape, batch["image"].dtype)  # (B,1,D,H,W)
        print("Mask:", batch["mask"].shape, batch["mask"].dtype)    # (B,1,D,H,W)
        break
