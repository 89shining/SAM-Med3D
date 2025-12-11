# test_nocrop.py
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
from scipy.ndimage import zoom

from testdataset import SAM3DTestDataset
from segment_anything import sam_model_registry3D


def restore_to_original_size(pred_128, original_shape, crop_h, crop_w,
                             z_min, z_max, pad_before, pad_after):
    """
    pred_128: (128,128,128) 预测 mask（二值）
    original_shape: 原图尺寸 (Z0,H0,W0)
    z_min, z_max: TestDataset 裁剪时记录的 GT Z 范围
    pad_before, pad_after: TestDataset Z-pad 时记录的 padding 数
    """

    Z0, H0, W0 = original_shape

    # ---------------------------------------------------
    # ① 先 undo XY resize: (128,128) → (160,128)
    # ---------------------------------------------------
    pred_unresize = zoom(pred_128,
                         (1, 160/128, 128/128),
                         order=0)  # mask 用 nearest

    # pred_unresize shape = (128,160,128)

    # ---------------------------------------------------
    # ② undo Z padding: 去掉 padding，恢复到 (Z_gt,160,128)
    # ---------------------------------------------------
    pred_unpad = pred_unresize[pad_before : 128 - pad_after]

    # pred_unpad shape = (Z_gt,160,128)

    # ---------------------------------------------------
    # ③ 将预测结果放回原图 z_min:z_max 区域
    # ---------------------------------------------------
    restored = np.zeros((Z0, crop_h, crop_w), dtype=np.uint8)

    Z_gt = pred_unpad.shape[0]
    assert Z_gt == (z_max - z_min + 1), "Z size mismatch!"

    restored[z_min:z_max+1] = pred_unpad  # 放回 GT 区域

    # ---------------------------------------------------
    # ④ 放回原图中心裁剪区域
    # ---------------------------------------------------
    final_mask = np.zeros(original_shape, dtype=np.uint8)

    cy, cx = H0 // 2, W0 // 2
    y1 = cy - crop_h // 2
    y2 = y1 + crop_h
    x1 = cx - crop_w // 2
    x2 = x1 + crop_w

    # 边界安全处理
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(H0, y2)
    x2 = min(W0, x2)

    final_mask[:, y1:y2, x1:x2] = restored[:, :y2-y1, :x2-x1]

    return final_mask


# ============================================================
#   ② 推理（完全与训练 forward 对齐）
# ============================================================
def run_inference(net, dataloader, device, save_dir):
    net.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():

        for batch in tqdm(dataloader, desc="Inference"):
            imgs = batch["image"].unsqueeze(0).to(device)  # (1,1,D,H,W)

            img_path = batch["img_path"]
            original_shape = batch["original_shape"]

            image_embeddings = net.image_encoder(imgs)

            batch_pred_masks = []
            for b in range(imgs.shape[0]):
                curr_emb = image_embeddings[b].unsqueeze(0)

                sparse_embeddings, dense_embeddings = net.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None
                )
                low_res_masks,_ = net.mask_decoder(
                    image_embeddings=curr_emb,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )

                pred_128 = torch.nn.functional.interpolate(
                    low_res_masks,
                    size=imgs.shape[-3:],  # (128,128,128)
                    mode="trilinear",
                    align_corners=False
                )[0, 0].sigmoid().cpu().numpy()

                pred_bin = (pred_128 > 0.5).astype(np.uint8)

            # ---------------- 保存 ----------------
            sitk_img = sitk.ReadImage(img_path)

            case_id = os.path.basename(img_path).replace("_0000.nii.gz", "")
            save_path = os.path.join(save_dir, f"{case_id}.nii.gz")

            restored = restore_to_original_size(
                pred_128=pred_bin,
                original_shape=original_shape,
                crop_h=160,
                crop_w=128,
                z_min=batch["z_min"],
                z_max=batch["z_max"],
                pad_before=batch["pad_before"],
                pad_after=batch["pad_after"]
            )

            # 转为 SimpleITK 图像
            restored_itk = sitk.GetImageFromArray(restored.astype(np.uint8))

            # 复制原图空间信息（非常重要）
            restored_itk.CopyInformation(sitk_img)

            # 保存
            sitk.WriteImage(restored_itk, save_path)
            print("Saved:", save_path)


# ============================================================
#   main
# ============================================================
if __name__ == "__main__":

    img_dir = "/home/wusi/SAM-Med3Ddata/Eso_CTV/dataset/test/imagesTs"
    mask_dir = "/home/wusi/SAM-Med3Ddata/Eso_CTV/dataset/test/labelsTs"
    save_dir = "/home/wusi/SAM-Med3Ddata/Eso_CTV/20251209/TestResult-SIcrop"

    dataset = SAM3DTestDataset(img_dir, mask_dir, crop_h=160, crop_w=128, target_size=(128,128,128))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    sam_checkpoint = "/home/wusi/SAM-Med3Ddata/Eso_CTV/20251209/TrainResult-SIcrop/fold_3/weights/best.pth"
    model_type = "vit_b_ori"

    device = torch.device("cuda:0")

    net = sam_model_registry3D[model_type](checkpoint=None)
    net.load_state_dict(torch.load(sam_checkpoint, map_location=device), strict=False)
    net.to(device)

    run_inference(net, dataloader, device, save_dir)