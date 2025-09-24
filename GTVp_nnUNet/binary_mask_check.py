"""
检查数据集mask是否为二值
"""

import os
import nibabel as nib
import numpy as np

def convert_to_binary_mask(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".nii.gz")]

    for fname in files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        # 读取nii.gz
        nii = nib.load(in_path)
        mask = nii.get_fdata().astype(np.int32)

        unique_vals = np.unique(mask)
        print(f"{fname} 原始标签值: {unique_vals}")

        if set(unique_vals).issubset({0, 1}):
            print(f"✅ {fname} 已经是二值，不做修改")
            bin_mask = mask
        else:
            print(f"⚠️ {fname} 转换为二值 (非0 → 1)")
            bin_mask = np.where(mask > 0, 1, 0)

        # 保存
        bin_img = nib.Nifti1Image(bin_mask.astype(np.uint8), nii.affine, nii.header)
        nib.save(bin_img, out_path)
        print(f"➡️ 已保存到 {out_path}\n")


if __name__ == "__main__":
    input_dir = r"C:\Users\dell\Desktop\SAM\GTVp_CTonly\20250809\nnUNet\Dataset001_GTVp\labelsTs"    # 你的nnUNet标签目录
    output_dir = r"C:\Users\dell\Desktop\labelsTs_binary"  # 输出目录
    convert_to_binary_mask(input_dir, output_dir)
