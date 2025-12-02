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

class Conv3D(nn.Module):
    """
    使用卷积 + stride=2 下采样：512→256 或 256→128。
    """
    def __init__(self, trainable=False):
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        if not trainable:
            for p in self.conv.parameters():
                p.requires_grad = False

            with torch.no_grad():
                self.conv.weight.fill_(1.0 / 27.0)  # 平均卷积

    def forward(self, x):
        # x: (1, D, H, W)
        x = x.unsqueeze(0)                # → (1,1,D,H,W)
        y = self.conv(x)
        return y.squeeze(0)              # → (1, D/2, H/2, W/2)


# ================================
#     两次下采样：512→256→128
# ================================
class TwoStageConvDownsample:
    def __init__(self, trainable=False):
        self.conv1 = Conv3D(trainable=trainable)   # 512→256
        self.conv2 = Conv3D(trainable=trainable)   # 256→128

        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

    def __call__(self, subject):
        img = subject['image'].data   # (1,D,H,W)
        mask = subject['label'].data  # (1,D,H,W)

        # ---- image 两次卷积下采样 ----
        img_256 = self.conv1(img)     # 512 → 256
        img_128 = self.conv2(img_256) # 256 → 128

        # ---- mask 两次池化下采样 ----
        mask_256 = self.pool1(mask.unsqueeze(0)).squeeze(0)
        mask_128 = self.pool2(mask_256.unsqueeze(0)).squeeze(0)

        subject['image'].set_data(img_128)
        subject['label'].set_data(mask_128)

        return subject


class SAM3DDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        """
        Args:
            img_dir (str): 图像文件夹路径 (imagesTr)
            mask_dir (str): 标签文件夹路径 (labelsTr)
        """
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
        assert len(self.img_paths) == len(self.mask_paths), "图像和mask数量不一致！"

        # 预处理
        self.transform = tio.Compose([
            tio.ToCanonical(),
            TwoStageConvDownsample(trainable=False),
            tio.ZNormalization(),
            tio.RandomFlip(axes=(0, 1, 2)),  # 数据增强：随机翻转
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

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.from_numpy(sitk.GetArrayFromImage(img_sitk)).unsqueeze(0)),
            label=tio.LabelMap(tensor=torch.from_numpy(sitk.GetArrayFromImage(mask_sitk)).unsqueeze(0))
        )

        #  应用 transform
        subject = self.transform(subject)

        img_tensor = subject['image'].data.float()  # (1,D,H,W)
        mask_tensor = subject['label'].data.float()  # (1,D,H,W)

        box = self.get_3d_box_from_mask(mask_tensor)

        return {
            "image": img_tensor,   # (1,D,H,W)
            "mask": mask_tensor,   # (1,D,H,W)
            "box": box             # (2,3)
        }

# ================== 测试 ==================
if __name__ == "__main__":
    img_dir = r"C:\Users\WS\Desktop\nnUNet-SAM\dataset\train\imagesTr"
    mask_dir = r"C:\Users\WS\Desktop\nnUNet-SAM\dataset\train\labelsTr"

    dataset = SAM3DDataset(img_dir, mask_dir, img_size=256)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        print("Image:", batch["image"].shape, batch["image"].dtype)  # (1,D,H,W)
        print("Mask:", batch["mask"].shape, batch["mask"].dtype)    # (1,D,H,W)
        print("Box:", batch["box"].shape, batch["box"])             # (2,3)
        break
