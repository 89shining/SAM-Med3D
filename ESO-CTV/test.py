import os
import sys

sys.path.append("/home/wusi/SAM-Med3D")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import logging
import numpy as np
import SimpleITK as sitk
from torch.utils.data import DataLoader
from tqdm import tqdm
from test_dataset import SAM3DTestDataset
from segment_anything import sam_model_registry3D


# ================== 工具函数 ==================
def remove_z_padding(pred_128, orig_Z):
    """
    pred_128: (128,128,128)
    orig_Z:   原始 Z 层数
    """
    return pred_128[:orig_Z, :, :]   # 因为前向 padding 是 pad Z 在尾部


def restore_to_original(pred_mask_np, resampled_img_path, original_img_path, save_path):
    """
    预测结果 -> (256³) -> 逆剪裁 -> resampled space -> 逆重采样 -> 原始 CT 空间
    """
    # Step1: 读取重采样后的 CT
    resampled_img = sitk.ReadImage(resampled_img_path)
    resampled_shape = sitk.GetArrayFromImage(resampled_img).shape

    # Step2: 逆剪裁
    pred_restored = inverse_crop_or_pad(pred_mask_np, resampled_shape)
    pred_itk = sitk.GetImageFromArray(pred_restored.astype(np.uint8))
    pred_itk.CopyInformation(resampled_img)  # 对齐重采样后的 CT header

    # Step3: 再 resample 回最原始 CT
    original_img = sitk.ReadImage(original_img_path)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(original_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    pred_original = resampler.Execute(pred_itk)

    # Step4: 保存
    sitk.WriteImage(pred_original, save_path)
    logging.info(f"Saved restored prediction: {save_path}")


# ================== 推理流程 ==================
def run_inference(net, dataloader, device, save_dir, ori_img_dir):
    net.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            imgs = batch['image'].to(device)  # (B,1,D,H,W)
            bbox = batch['box'].to(device)    # (B,2,3)

            # 先得到整个 batch 的 image_embeddings
            image_embeddings = net.image_encoder(imgs)

            batch_pred_masks = []
            for b in range(imgs.shape[0]):
                curr_embedding = image_embeddings[b].unsqueeze(0)
                curr_box = bbox[b].unsqueeze(0)

                sparse_embeddings, dense_embeddings = net.prompt_encoder(
                    points=None,
                    boxes=curr_box,
                    masks=None
                )
                low_res_masks, _ = net.mask_decoder(
                    image_embeddings=curr_embedding,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )

                # 上采样到 CropOrPad 后的尺寸
                true_masks = batch['mask'][b:b+1].to(device)
                pred_mask = torch.nn.functional.interpolate(
                    low_res_masks,
                    size=true_masks.shape[-3:],  # (D,H,W)
                    mode='trilinear',
                    align_corners=False
                )
                batch_pred_masks.append(pred_mask)

            pred_masks = torch.cat(batch_pred_masks, dim=0)  # (B,1,D,H,W)

            # ========= 恢复并保存 =========
            for b in range(imgs.shape[0]):
                pred_b = (pred_masks[b,0].sigmoid() > 0.5).cpu().numpy().astype(np.uint8)

                # 文件名
                case_id = os.path.basename(batch['img_path'][b]).replace(".nii.gz", "")
                resampled_img_path = batch['img_path'][b]  # Dataset 里的 1.5mm 图像
                original_img_path = os.path.join(ori_img_dir, f"{case_id}_0000.nii.gz")
                save_path = os.path.join(save_dir, f"{case_id}.nii.gz")

                restore_to_original(pred_b, resampled_img_path, original_img_path, save_path)


# ================== 主函数 ==================
if __name__ == '__main__':
    img_dir = "/home/wusi/SAM-Med3Ddata/dataset/test/imagesTs"       # 重采样(1.5mm)后的 img
    mask_dir = "/home/wusi/SAM-Med3Ddata/dataset/test/labelsTs"      # 重采样(1.5mm)后的 mask
    ori_img_dir = "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset001_GTVp/imagesTs"      # 最最原始 CT 路径
    save_dir = '/home/wusi/SAM-Med3Ddata/TestResult/GT_box'          # 保存预测结果

    dataset = SAM3DTestDataset(img_dir=img_dir, mask_dir=mask_dir, img_size=256)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    sam_checkpoint = "/home/wusi/SAM-Med3Ddata/TrainResult/GT_box/fold_5/weights/best.pth"
    model_type = "vit_b"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = sam_model_registry3D[model_type](checkpoint=None)
    state_dict = torch.load(sam_checkpoint, map_location=device)
    net.load_state_dict(state_dict, strict=False)
    net.to(device)

    run_inference(net, dataloader, device, save_dir, ori_img_dir)
