# -*- encoding: utf-8 -*-
'''
@File    :   prepare_data_from_nnUNet.py
@Time    :   2023/12/10 23:07:39
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   pre-process nnUNet-style dataset into SAM-Med3D-style
'''

import os.path as osp
import os
import json
import shutil
import nibabel as nib
from tqdm import tqdm
import torchio as tio
from glob import glob

# 重采样
def resample_nii(input_path: str, output_path: str, target_spacing: tuple = (1.5, 1.5, 1.5), n=None, reference_image=None, mode="linear"):
    """
    Resample a nii.gz file to a specified spacing using torchio.

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - output_path: Path to save the resampled .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    """
    
    # Load the nii.gz file using torchio
    subject = tio.Subject(
        img=tio.ScalarImage(input_path)
    )
    resampler = tio.Resample(target=target_spacing, image_interpolation=mode)
    resampled_subject = resampler(subject)
    
    if(n!=None):
        image = resampled_subject.img
        tensor_data = image.data
        if(isinstance(n, int)):
            n = [n]
        for ni in n:
            tensor_data[tensor_data == ni] = -1
        tensor_data[tensor_data != -1] = 0
        tensor_data[tensor_data != 0] = 1
        save_image = tio.ScalarImage(tensor=tensor_data, affine=image.affine)
        reference_size = reference_image.shape[1:]  # omitting the channel dimension
        cropper_or_padder = tio.CropOrPad(reference_size)
        save_image = cropper_or_padder(save_image)
    else:
        save_image = resampled_subject.img
    
    save_image.save(output_path)

# 路径定义
dataset_root = r"C:\Users\dell\Desktop\SAM\GTVp_CTonly\20250809\nnUNet"    # 原数据集根目录
dataset_list = [
    'Dataset001_GTVp',             # 原数据集名称
]

target_dir = r"C:\Users\dell\Desktop/nnUNet-SAM/dataset"    # 处理后的数据集保存地址


for dataset in dataset_list:
    dataset_dir = osp.join(dataset_root, dataset)   #  /nnUNet/Dataset001_GTVp
    meta_info = json.load(open(osp.join(dataset_dir, "dataset.json")))   # 读取dataset.json文件
    # print(meta_info['name'], meta_info['modality'])
    print(meta_info["channel_names"]["0"])   # 打印模态
    print(meta_info["numTraining"])   # 打印训练集数量

    # 打印分割类别（-1 除去background）
    num_classes = len(meta_info["labels"])-1
    print("num_classes:", num_classes, meta_info["labels"])

    # 重采样到1.5mm
    resample_dir = osp.join(dataset_dir, "imagesTr_1.5") 
    os.makedirs(resample_dir, exist_ok=True)
    for cls_name, idx in meta_info["labels"].items():
        cls_name = cls_name.replace(" ", "_")    # 取json里的类别标签：background GTVp
        idx = int(idx)    # 0 1
        dataset_name = dataset.split("_", maxsplit=1)[1]   # 取数据集名称下划线后部分：GTVp
        target_cls_dir = osp.join(target_dir, cls_name, dataset_name)  # /background/GTVp, /GTVp/GTVp
        target_img_dir = osp.join(target_cls_dir, "imagesTr")
        target_gt_dir = osp.join(target_cls_dir, "labelsTr")
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_gt_dir, exist_ok=True)

        images = sorted(glob(osp.join(dataset_dir, "imagesTr", "*.nii.gz")))
        labels = sorted(glob(osp.join(dataset_dir, "labelsTr", "*.nii.gz")))

        # image重采样
        for img, gt in zip(images, labels):
            resample_img = osp.join(resample_dir, osp.basename(img))
            if not osp.exists(resample_img):
                resample_nii(img, resample_img)
            img = resample_img

            # 去掉_0000后缀
            # img: GTVp_001_0000.nii,gz -> GTVp_001.nii.gz
            # gt：GTVp_001.nii.gz -> GTVp_001.nii.gz
            target_img_path = osp.join(target_img_dir, osp.basename(img).replace("_0000.nii.gz", ".nii.gz"))
            target_gt_path = osp.join(target_gt_dir, osp.basename(gt).replace("_0000.nii.gz", ".nii.gz"))

            # 读取gt
            gt_img = nib.load(gt)    
            spacing = tuple(gt_img.header['pixdim'][1:4])  #  x,y,z
            spacing_voxel = spacing[0] * spacing[1] * spacing[2]
            # 转为 numpy array (Z,Y,X)
            gt_arr = gt_img.get_fdata()
            # 把不属于当前idx的体素全部设为0
            gt_arr[gt_arr != idx] = 0
            # 剩下非0体素（当前idx）全部设为1
            gt_arr[gt_arr != 0] = 1
            # 统计值为1的所有体素体积
            volume = gt_arr.sum()*spacing_voxel
            # 如果体积小于10mm3，跳过（太小被认为是噪声）
            if(volume<10): 
                print("skip", target_img_path)
                continue

            # 读取重采样后的image  1.5mm spacing
            reference_image = tio.ScalarImage(img)
            # 特殊情况，将标签里的多类别编码合并
            if meta_info.get('name') == "kits23" and idx == 1:
                resample_nii(gt, target_gt_path, n=[1,2,3], reference_image=reference_image, mode="nearest")
            # 一般情况：提取第idx类，最近邻重采样与CT影像对齐
            else:
                resample_nii(gt, target_gt_path, n=idx, reference_image=reference_image, mode="nearest")
            # 将重采样的image转移到目标路径
            shutil.move(img, target_img_path)



