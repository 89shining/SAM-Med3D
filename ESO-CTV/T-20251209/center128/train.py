# 五折交叉验证 CTV训练
import os
import sys

import cv2
import numpy as np

sys.path.append("/home/wusi/SAM-Med3D")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import random
import logging
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from tqdm import tqdm
from monai.losses import DiceCELoss
from dataset import SAM3DDataset
from segment_anything import sam_model_registry3D

# 设置随机种子
manual_seed = int.from_bytes(os.urandom(4), 'little')
random.seed(manual_seed)
torch.manual_seed(manual_seed)

def visualize_sample(img, mask, box, save_path, alpha=0.4):
    """
    img : (D,H,W) numpy array
    mask: (D,H,W) numpy array, 0/1
    box : torch tensor (2,3)  [ [x1,y1,z1], [x2,y2,z2] ]
    save_path : 保存路径
    """

    # 转 numpy
    if hasattr(box, "cpu"):
        box = box.cpu().numpy()

    x1, y1, z1 = box[0].astype(int)
    x2, y2, z2 = box[1].astype(int)

    # 取 box 中心切片（Z轴）
    mid_z = int((z1 + z2) / 2)

    # 取该切片图像
    img2d = img[mid_z]      # (H,W)
    mask2d = mask[mid_z]    # (H,W)

    # ---- 灰度转可视化图 ----
    img_norm = cv2.normalize(img2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

    # ---- 生成 mask 叠加色（红色半透明） ----
    mask_color = np.zeros_like(img_rgb)
    mask_color[:, :, 2] = (mask2d * 255)  # 红色通道

    # cv2.addWeighted 做透明叠加
    overlay = cv2.addWeighted(img_rgb, 1.0, mask_color, alpha, 0)

    # ---- 绘制 box 投影 ----
    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),  # 绿色
        2
    )

    # ---- 保存 ----
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)

    return overlay


def train_one_fold(fold, train_idx, val_idx, dataset, net, device,
                   epochs, batch_size, lr, save_dir, clip_value=None):
    fold_dir = os.path.join(save_dir, f"fold_{fold + 1}")
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'runs'), exist_ok=True)

    # 日志记录
    logging.info(f'Auto-generated seed: {manual_seed}')

    # 直接从 dataset.img_paths 获取 ID
    train_ids = [os.path.basename(dataset.img_paths[i]) for i in train_idx]
    val_ids = [os.path.basename(dataset.img_paths[i]) for i in val_idx]


    with open(os.path.join(fold_dir, 'train_ids.txt'), 'w') as f:
        f.writelines(f"{id}\n" for id in train_ids)
    with open(os.path.join(fold_dir, 'val_ids.txt'), 'w') as f:
        f.writelines(f"{id}\n" for id in val_ids)

    # 同时保存到日志文件
    logging.info(f"Train IDs ({len(train_ids)} samples): {train_ids}")
    logging.info(f"Val IDs ({len(val_ids)} samples): {val_ids}")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

    writer = SummaryWriter(os.path.join(fold_dir, 'runs'))

    # 日志信息
    logging.info(f'''Starting training:
            Fold:            {fold + 1}
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {len(train_idx)}
            Validation size: {len(val_idx)}
            Device:          {device.type}
        ''')

    # 损失函数
    criterion = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

    # 学习率
    if isinstance(net, torch.nn.DataParallel):
        sam_model = net.module
    else:
        sam_model = net
    scalelr = lr
    optimizer = torch.optim.AdamW(
        [
            {"params": sam_model.image_encoder.parameters(), "lr": lr},  # 图像编码器
            {"params": sam_model.prompt_encoder.parameters(), "lr": lr},  # 提示编码器
            {"params": sam_model.mask_decoder.parameters(), "lr": lr},  # 掩码解码器
        ],
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )

    bestloss = float('inf')
    trainLoss, valLoss = [], []

    for epoch in range(epochs):
        net.train()
        train_epoch_loss = 0
        train_n_loss = 0
        with tqdm(total=len(train_loader), desc=f'[Train Fold {fold + 1}]', unit='batch', disable=True) as pbar:
            # 传入一个batch
            for batch_idx, batch in enumerate(train_loader):
                imgs = batch['image'].to(device)  # (B,1,D,H,W)
                true_masks = batch['mask'].to(device)  # (B,1,D,H,W)
                bbox = batch['box'].to(device)  # (B,2,3)

                #  训练可视化检查
                if epoch == 0 and batch_idx == 0:
                    img_np = imgs[0, 0].cpu().numpy()  # (D,H,W)
                    mask_np = true_masks[0, 0].cpu().numpy()
                    box_np = bbox[0]

                    out_path = os.path.join(fold_dir, f"debug_sample_epoch{epoch + 1}.png")
                    visualize_sample(img_np, mask_np, box_np, out_path)

                    print(f"[DEBUG] Saved visualization → {out_path}")

                # 先得到整个 batch 的 image_embeddings
                image_embeddings = net.image_encoder(imgs)

                batch_pred_masks = []
                for b in range(imgs.shape[0]):
                    curr_embedding = image_embeddings[b].unsqueeze(0)  # (1, C, D, H, W)
                    curr_box = bbox[b].unsqueeze(0)  # (1, 2, 3)

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

                    # # ========= 🔍 Debug 输出 ============
                    # print("\n[DEBUG SHAPES]")
                    # print("image_embeddings:", image_embeddings.shape)
                    # print("sparse embeddings:", sparse_embeddings.shape)
                    # print("dense embeddings:", dense_embeddings.shape)
                    # print("low_res_masks:", low_res_masks.shape)
                    # print("bbox:", bbox.shape, bbox[0])
                    # print("=================================\n")

                    # 上采样到原尺寸
                    pred_mask = torch.nn.functional.interpolate(
                        low_res_masks,
                        size=true_masks.shape[-3:],  # (D,H,W)
                        mode='trilinear',
                        align_corners=False
                    )
                    batch_pred_masks.append(pred_mask)

                # 拼回 batch
                pred_masks = torch.cat(batch_pred_masks, dim=0)  # (B, 1, D, H, W)

                train_loss = criterion(pred_masks, true_masks.float())

                # 返回当前batch的loss
                train_loss_batch = float(train_loss.item())
                # 当前epoch总loss
                train_epoch_loss += train_loss_batch
                train_n_loss += 1
                # 当前batch的loss反向传播
                optimizer.zero_grad()
                train_loss.backward()
                # 梯度裁剪：将梯度值限制在某个指定的范围内，防止梯度值过大导致训练不稳定
                # clip_value：梯度剪裁阈值
                if clip_value is not None:
                    nn.utils.clip_grad_value_(net.parameters(), clip_value)

                # 优化器参数更新
                optimizer.step()
                # torch.cuda.empty_cache()

                # 更新进度条右侧的附加信息:当前epoch的平均loss dice bce
                pbar.set_postfix({'TrainLoss': f"{train_epoch_loss / train_n_loss:.4f}"})
                # 更新进度条（进度条前进1步）
                pbar.update(1)

        train_meanLoss = train_epoch_loss / train_n_loss  # 当前epoch每个batch的平均损失
        trainLoss.append(train_meanLoss)
        writer.add_scalar('Loss/train_epoch_avg', train_meanLoss, epoch + 1)
        # torch.cuda.empty_cache()

        # Validation
        net.eval()
        val_epoch_loss = 0
        val_n_loss = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f'[Val Fold {fold + 1}]', unit='batch', disable=True) as pbar:
                # 传入一个batch
                for batch_idx, batch in enumerate(val_loader):
                    imgs = batch['image'].to(device)  # (B,1,D,H,W)
                    true_masks = batch['mask'].to(device)  # (B,1,D,H,W)
                    bbox = batch['box'].to(device)  # (B,2,3)

                    # 先得到整个 batch 的 image_embeddings
                    image_embeddings = net.image_encoder(imgs)

                    batch_pred_masks = []
                    for b in range(imgs.shape[0]):
                        curr_embedding = image_embeddings[b].unsqueeze(0)  # (1, C, D, H, W)
                        curr_box = bbox[b].unsqueeze(0)  # (1, 2, 3)

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
                        # 上采样到gt尺寸
                        pred_mask = torch.nn.functional.interpolate(
                            low_res_masks,
                            size=true_masks.shape[-3:],  # (D,H,W)
                            mode='trilinear',
                            align_corners=False
                        )
                        batch_pred_masks.append(pred_mask)

                    # 拼回 batch
                    pred_masks = torch.cat(batch_pred_masks, dim=0)  # (B, 1, D, H, W)

                    val_loss = criterion(pred_masks, true_masks.float())
                    # 返回当前batch的loss
                    val_loss_batch = float(val_loss.item())
                    # 当前epoch总loss
                    val_epoch_loss += val_loss_batch
                    # print("epoch_loss")
                    # print(epoch_loss)
                    val_n_loss += 1

                    # 更新进度条右侧的附加信息:当前epoch的平均loss
                    pbar.set_postfix({'ValLoss': f"{val_epoch_loss / val_n_loss:.4f}"})
                    # 更新进度条（进度条前进1步）
                    pbar.update(1)
                    # torch.cuda.empty_cache()

        val_meanLoss = val_epoch_loss / val_n_loss  # 当前epoch每个batch的平均损失
        valLoss.append(val_meanLoss)
        writer.add_scalar('Loss/Val_epoch_avg', val_meanLoss, epoch + 1)
        # torch.cuda.empty_cache()

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch + 1)
        logging.info(
            f'Epoch {epoch + 1}: Train Loss={trainLoss[-1]:.4f}, Val Loss={valLoss[-1]:.4f}, lr={current_lr:.8f}')

        scheduler.step(val_meanLoss)

        if bestloss > val_meanLoss:
            bestloss = val_meanLoss
            no_improve_epochs = 0
            torch.save(net.state_dict(),
                       os.path.join(fold_dir, 'weights') + f'/best.pth')  # 将最优模型权重保存为best.pth
            logging.info(f'Best model updated with loss={bestloss:.4f}')

    # 记录每个fold的最终结果
    with open(os.path.join(save_dir, 'summary.txt'), 'a') as f:
        f.write(f"Fold {fold + 1}: Best Val Loss = {bestloss:.4f}\n")

    # 绘制损失图
    plt.figure()
    plt.plot(range(1, len(trainLoss) + 1), trainLoss, label='Train Loss')
    plt.plot(range(1, len(valLoss) + 1), valLoss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold + 1} Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(fold_dir, 'loss_curve.jpg'))
    plt.close()
    writer.close()


if __name__ == '__main__':
    img_dir = "/home/wusi/SAM-Med3Ddata/Eso_CTV/dataset/train/imagesTr"
    mask_dir = "/home/wusi/SAM-Med3Ddata/Eso_CTV/dataset/train/labelsTr"
    save_dir = '/home/wusi/SAM-Med3Ddata/Eso_CTV/20251209/TrainResult'  # 训练结果保存文件夹
    os.makedirs(save_dir, exist_ok=True)

    dataset = SAM3DDataset(img_dir=img_dir, mask_dir=mask_dir, crop_h=160, crop_w=128, target_size=(128,128,128))
    all_image_paths = dataset.img_paths

    sam_checkpoint = "/home/wusi/SAM-Med3D/checkpoint/sam_med3d_turbo.pth"
    model_type = "vit_b_ori"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        # 只训练第4折和5折
        # if fold not in [2]:
        # continue

        # Logging setup
        log_path = os.path.join(save_dir, f'fold_{fold + 1}/train_fold{fold + 1}.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # 需要先移除已存在的 handler（否则重复 logging 会出错）
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, mode='w'),
                logging.StreamHandler(sys.stdout)]
        )
        logging.info(f"[Fold {fold + 1}] Logging initialized.")
        logging.info(f'Using device {device}')

        # 每次重新初始化网络
        net = sam_model_registry3D[model_type](checkpoint=None)
        state_dict = torch.load(sam_checkpoint, map_location=device)
        net.load_state_dict(state_dict, strict=False)
        logging.info(f"[Info] Loaded 3D SAM checkpoint from {sam_checkpoint} with strict=False.")
        net.to(device)

        # print("pixel_mean =", net.pixel_mean)
        # print("pixel_std =", net.pixel_std)
        # print("first layer weight =", net.image_encoder.patch_embed.proj.weight.shape)

        # 冻结图像编码器
        # for param in net.image_encoder.parameters():
        # param.requires_grad = False

        # 冻结解码器
        # for param in net.mask_decoder.parameters():
        # param.requires_grad = False

        trainable_params = [name for name, param in net.named_parameters() if param.requires_grad]
        logging.info(f"Trainable parameters ({len(trainable_params)}):")
        # print("Trainable parameters:")
        for name in trainable_params:
            logging.info(f"  {name}")
            # print(name)

        train_one_fold(fold, train_idx, val_idx, dataset, net, device,
                       epochs=200, batch_size=12, lr=0.0001, save_dir=save_dir)
        logging.info(f"Training Fold{fold + 1} completed.")

        torch.cuda.empty_cache()

    print("Five-fold cross-validation completed.")
