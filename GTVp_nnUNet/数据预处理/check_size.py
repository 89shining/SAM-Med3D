"""
检查文件尺寸
"""

import os
import SimpleITK as sitk
import glob

# 修改为你的 mask 文件夹路径
mask_dir = r"D:\SAM\GTVp_CTonly\20250809\testresults\nnUNet_3d"

mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))

for path in mask_paths:
    img = sitk.ReadImage(path)
    size = img.GetSize()        # (W, H, D) 注意 SimpleITK 返回的是 (x,y,z)
    spacing = img.GetSpacing()  # 体素间距
    print(f"{os.path.basename(path)} - Size: {size} (W,H,D), Spacing: {spacing}")
