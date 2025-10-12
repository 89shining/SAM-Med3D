"""
中间层基于nnUNet_mask的 3D框,上下界GT决定
GT提供上下界z，xy取并集：GT上下界层 + nnUNet中间层的xy
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
            # tio.RandomFlip(axes=(0, 1, 2)),  # 数据增强：随机翻转
            # tio.RandomNoise(mean=0, std=(0, 0.1)),
            # tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
        ])

    def __len__(self):
        return len(self.img_paths)

    def get_3d_box_predmiddle_gtedge(self, gt_mask, pred_mask):
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.squeeze(0).cpu().numpy()
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.squeeze(0).cpu().numpy()

        if np.count_nonzero(gt_mask) == 0:
            raise ValueError("GT 掩码为空，无法提取 z 范围！")

        coords_gt = np.argwhere(gt_mask > 0)
        z_min, _, _ = coords_gt.min(axis=0)
        z_max, _, _ = coords_gt.max(axis=0)

        # 提取 GT 的上下层（z_min, z_max）xy
        coords_top = np.argwhere(gt_mask[z_min] > 0)
        coords_bottom = np.argwhere(gt_mask[z_max] > 0)

        # 提取预测掩码的中间层（z_min+1 ~ z_max-1）
        coords_middle = np.argwhere(pred_mask[z_min + 1:z_max] > 0)  # 相对坐标

        # 合并三个来源的 xy 坐标
        all_coords = []
        if coords_top.shape[0] > 0:
            all_coords.append(coords_top)
        if coords_bottom.shape[0] > 0:
            all_coords.append(coords_bottom)
        if coords_middle.shape[0] > 0:
            all_coords.append(coords_middle[:, 1:])  # 去掉 z，只取 yx

        if len(all_coords) == 0:
            raise ValueError("GT 上下层 + nnUNet中间层全为空，无法生成框！")

        all_coords = np.vstack(all_coords)
        y_min, x_min = all_coords.min(axis=0)
        y_max, x_max = all_coords.max(axis=0)

        box = torch.tensor([
            [x_min, y_min, z_min],
            [x_max, y_max, z_max]
        ], dtype=torch.float32)

        return  box

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

        box = self.get_3d_box_predmiddle_gtedge(mask_tensor, box_tensor)

        return {
            "image": img_tensor,   # (1,D,H,W)
            "mask": mask_tensor,   # (1,D,H,W)
            "box": box,             # (2,3)
            "img_path": self.img_paths[idx]
        }

# ================== 测试 ==================
if __name__ == "__main__":
    img_dir = r"C:\Users\WS\Desktop\nnUNet-SAM/dataset/train/imagesTr"
    mask_dir = r"C:\Users\WS\Desktop\nnUNet-SAM/dataset/train/labelsTr"
    box_mask_dir = r"C:\Users\WS\Desktop\nnUNet-SAM/dataset/train/train_mask_box"
    dataset = SAM3DTestDataset(img_dir, mask_dir, box_mask_dir, img_size=256)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        print("Image:", batch["image"].shape, batch["image"].dtype)  # (1,D,H,W)
        print("Mask:", batch["mask"].shape, batch["mask"].dtype)    # (1,D,H,W)
        print("Box:", batch["box"].shape, batch["box"])             # (2,3)
        break
