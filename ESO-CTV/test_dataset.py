"""
image/CTV.nii.gz 512——Conv 3d —— 256 nii.gz——3d box
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


class ConvHW(nn.Module):
    """
    只对 H/W 下采样，Z 保持不变
    512→256→128，Z 不变
    """
    def __init__(self, trainable=False):
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),     # ★ Z 不动，只下采样 H/W
            padding=(0, 1, 1),
            bias=False
        )

        if not trainable:
            for p in self.conv.parameters():
                p.requires_grad = False
            with torch.no_grad():
                self.conv.weight.fill_(1.0 / 9.0)

    def forward(self, x):
        return self.conv(x)

class DownsampleAndPadZ:
    def __init__(self, target_z=128):
        self.conv1 = ConvHW()
        self.conv2 = ConvHW()
        self.target_z = target_z

    def __call__(self, subject):
        img = subject['image'].data  # (1, Z, H, W)
        mask = subject['label'].data

        # ---- 两次 H/W 下采样 ----
        img = self.conv1(img.unsqueeze(0)).squeeze(0)   # Z unchanged
        img = self.conv2(img.unsqueeze(0)).squeeze(0)

        mask = mask.unsqueeze(0)
        mask = F.max_pool3d(mask, kernel_size=(1,2,2), stride=(1,2,2))
        mask = F.max_pool3d(mask, kernel_size=(1,2,2), stride=(1,2,2))
        mask = mask.squeeze(0)

        # ---- Z padding to 128 ----
        z = img.shape[1]
        pad = self.target_z - z
        if pad > 0:
            img = F.pad(img, (0,0, 0,0, 0,pad))   # pad Z in front
            mask = F.pad(mask, (0,0, 0,0, 0,pad))

        subject['image'].set_data(img)
        subject['label'].set_data(mask)

        return subject

class SAM3DTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir,target_z=128):
        """
        Args:
            img_dir (str): 图像文件夹路径 (imagesTr)
            mask_dir (str): 标签文件夹路径 (labelsTr)
        """
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
        assert len(self.img_paths) == len(self.mask_paths), "图像和mask数量不一致！"
        self.target_z = target_z

        # 预处理
        self.transform = tio.Compose([
            tio.ToCanonical(),
            DownsampleAndPadZ(target_z),
            tio.ZNormalization(),
            # tio.RandomFlip(axes=(0, 1, 2)),  # 数据增强：随机翻转
            # 可选增强
            # tio.RandomNoise(mean=0, std=(0, 0.1)),
            # tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
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

        orig_Z = img_np.shape[0]

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
            "box": box,             # (2,3)
            "img_path": self.img_paths[idx],       # 用于恢复 CT header
            "orig_Z": orig_Z,           # ★ 恢复时用到
        }

# ================== 测试 ==================
# if __name__ == "__main__":
#     img_dir = r"C:\Users\dell\Desktop\dataset\train\imagesTr"
#     mask_dir = r"C:\Users\dell\Desktop\dataset\train\labelsTr"
#
#     dataset = SAM3DDataset(img_dir, mask_dir)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
#
#     for batch in dataloader:
#         print("Image:", batch["image"].shape, batch["image"].dtype)  # (B,1,D,H,W)
#         print("Mask:", batch["mask"].shape, batch["mask"].dtype)    # (B,1,D,H,W)
#         print("Box:", batch["box"].shape, batch["box"])             # (B,2,3)
#         break
