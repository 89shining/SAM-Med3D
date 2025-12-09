"""
检查mask像素分布，确定剪裁范围
"""

import os
import SimpleITK as sitk
import numpy as np

img_dir = "/home/wusi/SAM-Med3Ddata/Eso_CTV/dataset/train/imagesTr"
mask_dir = "/home/wusi/SAM-Med3Ddata/Eso_CTV/dataset/train/labelsTr"

image_files = sorted(os.listdir(img_dir))
mask_files  = sorted(os.listdir(mask_dir))

assert len(image_files) == len(mask_files)

max_dy_global = 0
max_dx_global = 0

for img_name, mask_name in zip(image_files, mask_files):

    img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(img_dir, img_name)))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_dir, mask_name)))

    D, H, W = img.shape

    # 图像中心
    cy = H // 2
    cx = W // 2

    # mask 前景坐标 (z,y,x)
    coords = np.argwhere(mask > 0)
    if coords.shape[0] == 0:
        continue

    ys = coords[:, 1]
    xs = coords[:, 2]

    # 计算距离中心的偏移
    dy = np.abs(ys - cy)
    dx = np.abs(xs - cx)

    max_dy = dy.max()
    max_dx = dx.max()

    max_dy_global = max(max_dy_global, max_dy)
    max_dx_global = max(max_dx_global, max_dx)

# print("\n===== Mask 中心像素偏移统计 =====")
# print(f"最大 dy (mask 到图像中心的最大垂直距离): {max_dy_global:.1f}")
# print(f"最大 dx (mask 到图像中心的最大水平距离): {max_dx_global:.1f}")

crop_h = int(max_dy_global * 2)
crop_w = int(max_dx_global * 2)

print("\n===== 最小中心裁剪窗口大小（无余量） =====")
print(f"最小 crop_h = {crop_h}")
print(f"最小 crop_w = {crop_w}")
print("（这是保证 mask 不被裁掉的严格最小尺寸）")
