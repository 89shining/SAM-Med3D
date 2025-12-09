"""
image/CTV.nii.gz 512——Crop160 128——Resize 128 128 128——3d box
"""

"""
基于GT的 3D框
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
            tio.ToCanonical(),
            tio.RandomFlip(axes=(1, 2)),  # 数据增强：随机翻转xy轴
        ])

    def __len__(self):
        return len(self.img_paths)

    def get_3d_box_from_mask(self, mask):
        # 如果是 torch.Tensor，转 numpy
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 4:  # [1,D,H,W] -> [D,H,W]
                mask = mask.squeeze(0)
            mask = mask.cpu().numpy()

        # 找非零体素坐标 (z,y,x)
        coords = np.argwhere(mask > 0)

        if coords.shape[0] == 0:
            raise ValueError("Empty mask detected! No foreground voxels found.")

        # z,y,x 最小最大值
        z_min, y_min, x_min = coords.min(axis=0).tolist()
        z_max, y_max, x_max = coords.max(axis=0).tolist()

        # 转成 (x,y,z) 格式，和 SAM3D Prompt 对齐
        box = torch.tensor([
            [x_min, y_min, z_min],
            [x_max, y_max, z_max]
        ], dtype=torch.float32)

        return box

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

        # 基于中心xy剪裁
        img_np, mask_np = center_crop_xy(img_np, mask_np, crop_h=self.crop_h, crop_w=self.crop_w)

        # 剪裁后mask体素
        after_crop_voxels = np.sum(mask_np > 0)
        if after_crop_voxels < orig_voxels:
            print(f"[Warning] Mask exceeds crop region for {self.img_paths[idx]}. "
                  f"Consider using larger crop_xy or using GT-based crop.")

        # resize
        img_np, mask_np = resize_3d(img_np, mask_np, target_size=self.target_size)


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

        box = self.get_3d_box_from_mask(mask_tensor)

        return {
            "image": img_tensor,   # (1,D,H,W)
            "mask": mask_tensor,   # (1,D,H,W)
            "box": box             # (2,3)
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
        print("Box:", batch["box"].shape, batch["box"])             # (B,2,3)
        break
