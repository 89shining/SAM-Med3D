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


# ============================================================
#   ① 还原为原始尺寸：128³ →  Z crop_w, crop_l → 原图尺寸
# ============================================================
import numpy as np
from scipy.ndimage import zoom

def restore_to_original_size(pred_128, original_shape, crop_h, crop_w):
    """
    pred_128: 模型预测的 mask，shape = (128,128,128) 或 (D,H,W)
    original_shape: 原图 shape，例如 (Z0, H0, W0)
    crop_h, crop_w: 训练/测试使用的剪裁尺寸（160×128）

    return: restored_mask, shape=(Z0, H0, W0)
    """
    Z0, H0, W0 = original_shape
    D1, H1, W1 = pred_128.shape   # 128,128,128

    # ------------------------
    # ① 先还原到 crop 的大小 (Z_crop × 160 × 128)
    zoom_factors = (
        Z0 / D1,           # Z 方向比例
        crop_h / H1,       # 128 → 160
        crop_w / W1        # 128 → 128
    )

    pred_crop = zoom(pred_128, zoom_factors, order=1)

    # pred_crop shape should be: (Z0, 160, 128)

    # ------------------------
    # ② 创建原图大小空 mask
    # ------------------------
    restored = np.zeros(original_shape, dtype=np.float32)

    # ------------------------
    # ③ 计算中心裁剪对应的 XY 区域并放回原图
    # ------------------------
    cy, cx = H0 // 2, W0 // 2
    y1 = cy - crop_h // 2
    y2 = y1 + crop_h
    x1 = cx - crop_w // 2
    x2 = x1 + crop_w

    # 防止越界
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(H0, y2)
    x2 = min(W0, x2)

    # ------------------------
    # ④ 把 pred_crop 放回原图
    # ------------------------
    restored[:, y1:y2, x1:x2] = pred_crop

    return restored


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
                crop_w=128
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
    save_dir = "/home/wusi/SAM-Med3Ddata/Eso_CTV/20251209/TestResult-noprompt"

    dataset = SAM3DTestDataset(img_dir, mask_dir, crop_h=160, crop_w=128, target_size=(128,128,128))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    sam_checkpoint = "/home/wusi/SAM-Med3Ddata/Eso_CTV/20251209/TrainResult-noprompt/fold_3/weights/best.pth"
    model_type = "vit_b_ori"

    device = torch.device("cuda:0")

    net = sam_model_registry3D[model_type](checkpoint=None)
    net.load_state_dict(torch.load(sam_checkpoint, map_location=device), strict=False)
    net.to(device)

    run_inference(net, dataloader, device, save_dir)