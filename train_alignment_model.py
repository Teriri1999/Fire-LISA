import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import random

from model.alignment_model import GeoAlignmentModel
from utils.wildfire_dataset import WildfireDataset

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Step-1: GeoAdapter Alignment Training (Wildfire)")

    # ---- 数据 ----
    parser.add_argument(
        "--data_roots", type=str, nargs="+",
        default=[
            "/home/x/u/xuranh/wildfire-dataset-CA-2022",
            "/home/x/u/xuranh/wildfire-dataset-CA-2023",
            "/home/x/u/xuranh/wildfire-dataset-CA-2024",
        ],
        help="一个或多个年份数据集根目录",
    )
    parser.add_argument("--crop_size",  type=int, default=256,
                        help="随机裁剪尺寸（像素）")
    parser.add_argument("--samples_per_epoch", type=int, default=8000)

    # ---- 模型 ----
    parser.add_argument("--save_dir",   type=str, default="./alignment_weight")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--geo_model",  type=str, default="convnext_tiny")

    # ---- 训练超参 ----
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--num_workers",  type=int,   default=8)
    parser.add_argument("--seed",         type=int,   default=42)

    # ---- GPU ----
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7,8,9",
                        help="可见 GPU 编号，如 '0,1,2'")

    # ---- 断点续训 ----
    parser.add_argument("--resume",      type=str, default="")
    parser.add_argument("--start_epoch", type=int, default=0)

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def step1_collate_fn(batch):
    final_input_images = torch.stack([item[1] for item in batch])  # [B, 10, H, W]
    image_clips        = torch.stack([item[2] for item in batch])  # [B,  3, 224, 224]
    return [
        None,
        final_input_images,
        image_clips,
        None, None, None, None, None, None,
    ]


def get_adapter(model):
    """兼容 DataParallel 和单卡，统一返回 geo_adapter。"""
    return model.module.geo_adapter if isinstance(model, nn.DataParallel) else model.geo_adapter


def main():
    args = parse_args()

    # ---- GPU 设置 ----
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    n_gpus = len(args.gpus.split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Step-1 Wildfire Alignment Training")
    print(f"  GPUs : {args.gpus} ({n_gpus} devices)")
    print(f"  Data : {args.data_roots}")

    # ---- Dataset ----
    dataset = WildfireDataset(
        data_roots=args.data_roots,
        vision_tower=args.clip_model,
        crop_size=args.crop_size,
        samples_per_epoch=args.samples_per_epoch,
        augment=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=step1_collate_fn,
    )

    # ---- Model ----
    model = GeoAlignmentModel(
        clip_model_name=args.clip_model,
        geo_model_name=args.geo_model,
    ).to(device)

    if n_gpus > 1 and torch.cuda.device_count() > 1:
        print(f"  Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # ---- Resume ----
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        get_adapter(model).load_state_dict(ckpt)
        print("Checkpoint loaded.")

    # ---- Optimizer ----
    optimizer = optim.AdamW(
        [p for p in get_adapter(model).parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 恢复调度器状态
    for _ in range(args.start_epoch):
        scheduler.step()

    scaler = GradScaler()

    print(f"Start training from epoch {args.start_epoch + 1} to {args.epochs} ...")

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            final_input_image = batch[1].to(device, non_blocking=True)  # [B, 10, H, W]
            image_clip        = batch[2].to(device, non_blocking=True)  # [B,  3, 224, 224]

            # S2 pre+post = ch 4:10，6 通道，Teacher GeoEncoder in_chans=6
            geo_input = final_input_image[:, 4:10, :, :]

            optimizer.zero_grad()
            with autocast():
                loss = model(image_clip, geo_input)

            # DataParallel 返回多 GPU 上的 loss 张量，取均值
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "lr":   f"{scheduler.get_last_lr()[0]:.2e}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} | avg_loss={avg_loss:.5f}")

        save_path = os.path.join(args.save_dir, f"geo_adapter_epoch_{epoch + 1}.pth")
        torch.save(get_adapter(model).state_dict(), save_path)
        print(f"Saved: {save_path}")

        scheduler.step()


if __name__ == "__main__":
    main()
