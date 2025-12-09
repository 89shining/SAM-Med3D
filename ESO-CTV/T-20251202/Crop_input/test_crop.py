# test_crop.py

import os
import sys
sys.path.append("/home/wusi/SAM-Med3D")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import zoom

from testdataset_crop import SAM3DTestDataset   # ← 你确认无误的版本
from segment_anything import sam_model_registry3D


# ============================================================
#   恢复到原图大小 (crop → zoom back → pad to original)
# ============================================================
def restore_mask_to_original(pred_128, crop_shape, z_min, original_shape):
    crop_Z, H_crop, W_crop = crop_shape
    Z0, H0, W0 = original_shape

    # ---- Step 1: 128 → crop_size ----
    zoom_factor = [
        crop_Z / pred_128.shape[0],
        H_crop / pred_128.shape[1],
        W_crop / pred_128.shape[2]
    ]
    mask_crop = zoom(pred_128, zoom_factor, order=0)

    # ---- Step 2: pad 回原图 ----
    final_mask = np.zeros(original_shape, dtype=np.uint8)
    z_max = z_min + crop_Z
    final_mask[z_min:z_max] = mask_crop

    return final_mask


# ============================================================
#   推理主流程（与你 training forward 一致）
# ============================================================
def run_inference_crop(net, dataloader, device, save_dir):
    net.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):

            # Dataset 输出字段：
            # image, mask, box, img_path, original_shape, crop_shape, z_min

            imgs = batch["image"].unsqueeze(0).to(device)  # (1,1,D,H,W)
            box = batch["box"].unsqueeze(0).to(device)

            z_min          = int(batch["z_min"])
            crop_shape     = batch["crop_shape"]      # (Zc,H,W)
            original_shape = batch["original_shape"]  # (Z,H,W)
            img_path       = batch["img_path"]

            # ============================================================
            #   ① forward（与你训练完全一致）
            # ============================================================
            image_embeddings = net.image_encoder(imgs)
            curr_emb = image_embeddings[0].unsqueeze(0)
            curr_box = box[0].unsqueeze(0)

            sparse_embeddings, dense_embeddings = net.prompt_encoder(
                points=None,
                boxes=curr_box,
                masks=None
            )

            low_res_masks, _ = net.mask_decoder(
                image_embeddings=curr_emb,
                image_pe=net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )

            # 上采样回 128×128×128
            pred_128 = torch.nn.functional.interpolate(
                low_res_masks,
                size=imgs.shape[-3:],
                mode='trilinear',
                align_corners=False
            )[0, 0].sigmoid().cpu().numpy()

            pred_bin = (pred_128 > 0.5).astype(np.uint8)

            # ============================================================
            #   ② 还原到原图尺寸
            # ============================================================
            restored_mask = restore_mask_to_original(
                pred_bin,
                crop_shape=crop_shape,
                z_min=z_min,
                original_shape=original_shape
            )

            # ============================================================
            #   ③ 写回 nii.gz
            # ============================================================
            original_sitk = sitk.ReadImage(img_path)
            pred_sitk = sitk.GetImageFromArray(restored_mask.astype(np.uint8))

            pred_sitk.SetSpacing(original_sitk.GetSpacing())
            pred_sitk.SetOrigin(original_sitk.GetOrigin())
            pred_sitk.SetDirection(original_sitk.GetDirection())

            base = os.path.basename(img_path)  # e.g. "CTV_014_0000.nii.gz"
            case_id = base.split('_')[1]  # "014"
            save_name = f"CTV_{case_id}.nii.gz"  # "CTV_014.nii.gz"

            save_path = os.path.join(save_dir, save_name)
            sitk.WriteImage(pred_sitk, save_path)

            print(f"Saved: {save_path}")


# ============================================================
#   main
# ============================================================
if __name__ == "__main__":

    img_dir = "/home/wusi/SAM-Med3Ddata/Eso_CTV/dataset/test/imagesTs"
    mask_dir = "/home/wusi/SAM-Med3Ddata/Eso_CTV/dataset/test/labelsTs"
    save_dir = "/home/wusi/SAM-Med3Ddata/Eso_CTV/TestResult_crop"

    checkpoint = "/home/wusi/SAM-Med3Ddata/Eso_CTV/TrainResult_crop/fold_1/weights/best.pth"
    model_type = "vit_b_ori"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load model ----
    net = sam_model_registry3D[model_type](checkpoint=None)
    net.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    net.to(device)

    # ---- load dataset（必须加 collate_fn）----
    dataset = SAM3DTestDataset(img_dir, mask_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0]
    )

    # ---- run inference ----
    run_inference_crop(net, dataloader, device, save_dir)
