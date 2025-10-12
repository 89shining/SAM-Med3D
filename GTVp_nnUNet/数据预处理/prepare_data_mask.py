import os
import nibabel as nib
import torchio as tio
from glob import glob

def resample_mask(input_path, output_path, ref_img_path, target_spacing=(1.5,1.5,1.5), min_volume=10):
    """
    将外部 mask 重采样并对齐到对应的 CT 图像。

    Args:
        input_path (str): mask 输入路径
        output_path (str): 保存路径
        ref_img_path (str): 对应的 CT 图像路径
        target_spacing (tuple): 目标 spacing
        min_volume (float): 体积阈值 (mm³)
    """
    # 读取mask
    mask_img = nib.load(input_path)
    spacing = tuple(mask_img.header['pixdim'][1:4])  # 原始 spacing
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    mask_arr = mask_img.get_fdata()

    # 二值化
    mask_arr[mask_arr != 0] = 1
    volume = mask_arr.sum() * voxel_volume
    if volume < min_volume:
        print(f"Skip {os.path.basename(input_path)} (volume={volume:.2f} mm³)")
        return False

    # TorchIO subject
    subject = tio.Subject(label=tio.LabelMap(input_path))
    resampler = tio.Resample(target=target_spacing, image_interpolation="nearest")
    resampled = resampler(subject)

    # 加载对应的参考 CT
    ref_img = tio.ScalarImage(ref_img_path)

    # 保证和 reference_image 对齐
    ref_size = ref_img.shape[1:]  # (D, H, W)
    crop_or_pad = tio.CropOrPad(ref_size)
    aligned = crop_or_pad(resampled)

    # 保存
    aligned.label.save(output_path)
    print(f"Saved: {output_path}")
    return True


if __name__ == "__main__":
    mask_dir = r"C:\Users\WS\Desktop\nnUNet-SAM\box\test_box"        # 外部 mask 文件夹
    ref_img_dir = r"C:\Users\WS\Desktop/nnUNet-SAM/dataset/test\labelsTs"  # 对应 CT 文件夹
    save_dir = r"C:\Users\WS\Desktop\nnUNet-SAM\dataset\test\test_mask_box"
    os.makedirs(save_dir, exist_ok=True)

    for mask_path in glob(os.path.join(mask_dir, "*.nii.gz")):
        filename = os.path.basename(mask_path)
        ref_img_path = os.path.join(ref_img_dir, filename)  # 找同名 CT
        if not os.path.exists(ref_img_path):
            print(f"Reference image not found for {filename}, skip.")
            continue

        save_path = os.path.join(save_dir, filename)
        resample_mask(mask_path, save_path, ref_img_path)
