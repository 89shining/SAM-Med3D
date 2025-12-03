# test_dataset_crop.py

import os
import glob
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from scipy.ndimage import zoom
from torch.utils.data import DataLoader


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

        img_resized = zoom(img_np, zoom_factor, order=3)
        mask_resized = zoom(mask_np, zoom_factor, order=0)
        return img_resized, mask_resized


# -------------------------------
#   ② GPT crop（Z方向裁剪）
# -------------------------------
def crop_by_mask(img_np, mask_np):
    coords = np.where(mask_np > 0)

    if len(coords[0]) == 0:
        return img_np, mask_np, 0, img_np.shape[0]-1

    z_min = coords[0].min()
    z_max = coords[0].max()

    img_crop = img_np[z_min:z_max + 1]
    mask_crop = mask_np[z_min:z_max + 1]

    return img_crop, mask_crop, z_min, z_max


# -------------------------------
#   ③ Test Dataset（与训练完全一致）
# -------------------------------
class SAM3DTestDataset(Dataset):
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
            return torch.tensor([[0,0,0],[0,0,0]], dtype=torch.float32)

        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        return torch.tensor([[x_min, y_min, z_min],
                             [x_max, y_max, z_max]], dtype=torch.float32)

    def __getitem__(self, idx):
        img_sitk = sitk.ReadImage(self.img_paths[idx])
        mask_sitk = sitk.ReadImage(self.mask_paths[idx])

        img_np = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        mask_np = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)

        original_shape = tuple(img_np.shape)   # (Z,H,W)

        # ---------- ① crop ----------
        img_crop, mask_crop, z_min, z_max = crop_by_mask(img_np, mask_np)
        crop_shape = tuple(img_crop.shape)   # (Z_crop, H, W)

        # -----------② zoom -----------
        img_resized, mask_resized = self.resize(img_crop, mask_crop)

        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()

        box = self.get_3d_box_from_mask(mask_tensor)

        z_min = int(z_min)
        z_max = int(z_max)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "box": box,
            "img_path": self.img_paths[idx],
            "original_shape": original_shape,   # (Z,H,W)
            "crop_shape": crop_shape,           # (Z_crop,H,W)
            "z_min": z_min,
            "z_max": z_max
        }


# -------------------------------
#   ④ 测试
# -------------------------------
# if __name__ == "__main__":
#     img_dir = "/home/wusi/SAM-Med3Ddata/Eso_CTV/dataset/test/imagesTs"
#     mask_dir = "/home/wusi/SAM-Med3Ddata/Eso_CTV/dataset/test/labelsTs"
#     dataset = SAM3DTestDataset(img_dir, mask_dir)
#     loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
#
#     for batch in loader:
#         print("image:", batch["image"].shape)  # (1,1,128,128,128)
#         print("mask:", batch["mask"].shape)  # (1,1,128,128,128)
#         print("box:", batch["box"])
#         print("original_shape:", batch["original_shape"])
#         print("crop_shape:", batch["crop_shape"])  # 必须是 3 个值 (Z_crop,H,W)
#         print("z_min:", batch["z_min"])
#         print("z_max:", batch["z_max"])
#         print("img_path:", batch["img_path"])
#         break
