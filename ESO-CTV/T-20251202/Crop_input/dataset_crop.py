"""
只保留GT切片体积
"""

import os
import glob
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import torch.nn.functional as F
from scipy.ndimage import zoom


# -------------------------------
#   ① zoom 缩放模块（保持不变）
# -------------------------------
class ResizeWithZoom:
    def __init__(self, target_size=(128, 128, 128)):
        self.target_z, self.target_h, self.target_w = target_size

    def __call__(self, img_np, mask_np):
        z, h, w = img_np.shape

        zoom_factor = [
            self.target_z / z,
            self.target_h / h,
            self.target_w / w
        ]

        # zoom
        img_resized = zoom(img_np, zoom_factor, order=3)
        mask_resized = zoom(mask_np, zoom_factor, order=0)

        return img_resized, mask_resized


# -------------------------------
#   ② GPT crop（Z方向裁剪）
# -------------------------------
def crop_by_mask(img_np, mask_np):
    """
    输入：(Z,H,W) or (H,W,Z)？我们做统一。
    sitk 读取后是 (Z,H,W)
    """

    # 找 mask>0 的 slice 范围
    coords = np.where(mask_np > 0)

    if len(coords[0]) == 0:
        # mask 空 → 不裁剪
        return img_np, mask_np

    z_min = coords[0].min()
    z_max = coords[0].max()

    # 裁剪 Z
    img_crop = img_np[z_min:z_max + 1]
    mask_crop = mask_np[z_min:z_max + 1]

    return img_crop, mask_crop


# -------------------------------
#   ③ Dataset 整合
# -------------------------------
class SAM3DDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))

        assert len(self.img_paths) == len(self.mask_paths)

        self.resize = ResizeWithZoom(target_size=(128, 128, 128))

    def __len__(self):
        return len(self.img_paths)

    def get_3d_box_from_mask(self, mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze(0).cpu().numpy()

        coords = np.argwhere(mask > 0)
        if coords.shape[0] == 0:
            return torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float32)

        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        return torch.tensor([[x_min, y_min, z_min],
                             [x_max, y_max, z_max]], dtype=torch.float32)

    def __getitem__(self, idx):
        # ---- 读取 ----
        img_sitk = sitk.ReadImage(self.img_paths[idx])
        mask_sitk = sitk.ReadImage(self.mask_paths[idx])

        img_np = sitk.GetArrayFromImage(img_sitk).astype(np.float32)   # (Z,H,W)
        mask_np = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)

        # -------------------------------
        # ① 先按 GTV 裁剪（Z 方向）
        # -------------------------------
        img_np, mask_np = crop_by_mask(img_np, mask_np)
        print(img_np.shape, mask_np.shape)

        # -------------------------------
        # ② zoom 缩放到 128×128×128
        # -------------------------------
        img_np, mask_np = self.resize(img_np, mask_np)

        # -------------------------------
        # ③ 转成 tensor 给模型
        # -------------------------------
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()   # (1,D,H,W)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()

        # -------------------------------
        # ④ 生成 3D box
        # -------------------------------
        box = self.get_3d_box_from_mask(mask_tensor)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "box": box
        }


# -------------------------------
#   ④ 测试
# -------------------------------
# if __name__ == "__main__":
#     img_dir = r"C:\Users\dell\Desktop\dataset\All_input\imagesTr"
#     mask_dir = r"C:\Users\dell\Desktop\dataset\All_input\labelsTr"
#
#     dataset = SAM3DDataset(img_dir, mask_dir)
#     loader = DataLoader(dataset, batch_size=1, shuffle=True)
#
#     for batch in loader:
#         print(batch["image"].shape)   # (1,1,128,128,128)
#         print(batch["mask"].shape)
#         print(batch["box"])
#         break
