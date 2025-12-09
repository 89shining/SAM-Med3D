"""
基于nnUNet_mask的 3D框
"""

import os
import glob
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
import torchio as tio

class SAM3DTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, box_mask_dir, img_size):
        """
        Args:
            img_dir (str): 图像文件夹路径 (imagesTr)
            mask_dir (str): 标签文件夹路径 (labelsTr)
        """
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
        self.box_mask_paths = sorted(glob.glob(os.path.join(box_mask_dir, "*.nii.gz")))
        assert len(self.img_paths) == len(self.mask_paths) == len(self.box_mask_paths), "图像和mask数量不一致！"

        # 预处理
        self.transform = tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(target_shape=(img_size, img_size, img_size),
            ),
            tio.ZNormalization(masking_method=lambda x: x > 0),  # 仅对前景归一化
            # 可选增强
            # tio.RandomNoise(mean=0, std=(0, 0.1)),
            # tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
        ])

    def __len__(self):
        return len(self.img_paths)

    def get_3d_box_from_mask(self, mask):
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
        box_mask_sitk = sitk.ReadImage(self.box_mask_paths[idx])

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.from_numpy(sitk.GetArrayFromImage(img_sitk)).unsqueeze(0)),
            label=tio.LabelMap(tensor=torch.from_numpy(sitk.GetArrayFromImage(mask_sitk)).unsqueeze(0)),
            box_mask = tio.LabelMap(tensor=torch.from_numpy(sitk.GetArrayFromImage(box_mask_sitk)).unsqueeze(0))
        )

        #  应用 transform
        subject = self.transform(subject)

        img_tensor = subject['image'].data.float()  # (1,D,H,W)
        mask_tensor = subject['label'].data.float()  # (1,D,H,W)
        box_tensor = subject['box_mask'].data.float()

        box = self.get_3d_box_from_mask(box_tensor)

        return {
            "image": img_tensor,   # (1,D,H,W)
            "mask": mask_tensor,   # (1,D,H,W)
            "box": box,             # (2,3)
            "img_path": self.img_paths[idx]
        }

# ================== 测试 ==================
if __name__ == "__main__":
    img_dir = r"C:\Users\WS\Desktop\nnUNet-SAM/dataset/All_input/imagesTr"
    mask_dir = r"C:\Users\WS\Desktop\nnUNet-SAM/dataset/All_input/labelsTr"
    box_mask_dir = r"C:\Users\WS\Desktop\nnUNet-SAM/dataset/All_input/train_mask_box"
    dataset = SAM3DTestDataset(img_dir, mask_dir, box_mask_dir, img_size=256)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        print("Image:", batch["image"].shape, batch["image"].dtype)  # (1,D,H,W)
        print("Mask:", batch["mask"].shape, batch["mask"].dtype)    # (1,D,H,W)
        print("Box:", batch["box"].shape, batch["box"])             # (2,3)
        break
