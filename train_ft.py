"""
Direct SAR fine-tuning of vanilla LISA-7B on wildfire dataset.
No alignment pre-training (no geo_tower / geo_adapter).
SAR is converted to pseudo-RGB → standard CLIP + SAM pipeline.

Run on autodl from inside the LISA repo directory:
    cd /root/autodl-tmp/LISA
    deepspeed --include localhost:0,1,2,3 train_lisa_sar_direct.py \
        --version /root/autodl-tmp/LISA-7B \
        --vision_pretrained /root/autodl-tmp/sam_vit_h_4b8939.pth \
        --data_roots /root/autodl-tmp/wildfire-dataset-CA-2022 \
                     /root/autodl-tmp/wildfire-dataset-CA-2023 \
                     /root/autodl-tmp/wildfire-dataset-CA-2024 \
        --exp_name lisa_sar_direct \
        --epochs 10 \
        --precision bf16
"""

import argparse
import os
import random
import re
import shutil
import sys
import time
import warnings
from functools import partial

import cv2
import deepspeed
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import transformers
from osgeo import gdal
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPImageProcessor

warnings.filterwarnings("ignore")
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# LISA repo imports (run from /root/autodl-tmp/LISA)
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from utils.utils import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN,
    AverageMeter, ProgressMeter, Summary, dict_to_cuda, intersectionAndUnionGPU,
)

# ---------------------------------------------------------------------------
# SAR → RGB helpers
# ---------------------------------------------------------------------------

_SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53])
_SAM_PIXEL_STD  = torch.tensor([58.395, 57.12, 57.375])

_FIRE_ID_RE = re.compile(r'^(CA_\d{4}_[A-Z]+_\d+)')
_MASK_RE    = re.compile(r'^(CA_\d{4}_[A-Z]+_\d+)_mask\.tif$')
S1_MODALITIES = ['S1_HS', 'S1_AG']


def _read_bands(path, band_indices=None):
    ds = gdal.Open(path)
    if ds is None:
        raise IOError(f"GDAL cannot open: {path}")
    if band_indices is None:
        band_indices = list(range(1, ds.RasterCount + 1))
    bands = [ds.GetRasterBand(bi).ReadAsArray().astype(np.float32) for bi in band_indices]
    ds = None
    arr = np.stack(bands, axis=0)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _percentile_normalize(arr):
    out = np.empty_like(arr)
    for c in range(arr.shape[0]):
        ch = arr[c]
        valid = ch[np.isfinite(ch)]
        if valid.size == 0:
            out[c] = 0.0
            continue
        p2, p98 = np.percentile(valid, [2, 98])
        rng = p98 - p2
        out[c] = np.clip((ch - p2) / rng, 0.0, 1.0) if rng > 1e-6 else np.zeros_like(ch)
    return out


def _list_by_event(directory):
    result = {}
    if not os.path.isdir(directory):
        return result
    for fname in os.listdir(directory):
        if not fname.lower().endswith('.tif'):
            continue
        m = _FIRE_ID_RE.match(fname)
        if m:
            result.setdefault(m.group(1), []).append(os.path.join(directory, fname))
    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_WILDFIRE_QUESTIONS = [
    "Where is the burned area in this pre- and post-fire satellite image?",
    "Please segment the wildfire burn scar in this remote sensing image.",
    "Identify the fire-damaged region shown in this change-detection image.",
    "Locate the area affected by wildfire in this satellite imagery.",
    "Which region shows evidence of fire damage in this image?",
]

_WILDFIRE_ANSWERS = [
    "The burned area is [SEG].",
    "The wildfire burn scar is located at [SEG].",
    "The fire-damaged region is [SEG].",
    "The area affected by wildfire is shown by [SEG].",
]


class LISASARDataset(Dataset):
    """
    Direct SAR → LISA dataset.

    images       [3, 1024, 1024]: SAM-normalised pseudo-RGB
    images_clip  [3, 224, 224]:   CLIP-preprocessed pseudo-RGB

    Pseudo-RGB composite (same as eval script):
        R = S1_pre  VV
        G = S1_post VV
        B = S1_pre  VH
    """

    def __init__(
        self,
        data_roots,
        tokenizer,
        vision_tower,
        samples_per_epoch=10000,
        precision="fp32",
        image_size=1024,
        crop_size=512,
        min_crop_size=256,
    ):
        self.tokenizer         = tokenizer
        self.precision         = precision
        self.img_size          = image_size
        self.crop_size         = crop_size
        self.min_crop_size     = min_crop_size
        self.samples_per_epoch = samples_per_epoch
        self.clip_processor    = CLIPImageProcessor.from_pretrained(vision_tower)

        self.events = []
        for root in data_roots:
            s1_pre, s1_post = {}, {}
            for mod in S1_MODALITIES:
                for eid, files in _list_by_event(os.path.join(root, mod, 'pre')).items():
                    s1_pre.setdefault(eid, []).extend(files)
                for eid, files in _list_by_event(os.path.join(root, mod, 'post')).items():
                    s1_post.setdefault(eid, []).extend(files)

            mask_map = {}
            mask_dir = os.path.join(root, 'mask')
            if os.path.isdir(mask_dir):
                for fname in os.listdir(mask_dir):
                    m = _MASK_RE.match(fname)
                    if m:
                        mask_map[m.group(1)] = os.path.join(mask_dir, fname)

            valid_ids = set(s1_pre) & set(s1_post) & set(mask_map)
            for eid in sorted(valid_ids):
                self.events.append({
                    's1_pre':  s1_pre[eid],
                    's1_post': s1_post[eid],
                    'mask':    mask_map[eid],
                    'id':      eid,
                })

        print(f"[LISASARDataset] {len(self.events)} events from {len(data_roots)} root(s)")
        if not self.events:
            raise RuntimeError("No valid fire events found. Check data_roots.")

    def __len__(self):
        return self.samples_per_epoch

    # -- spatial helpers --

    def _ensure_min_size(self, arr, cs):
        _, h, w = arr.shape
        ph, pw = max(0, cs - h), max(0, cs - w)
        return np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode='reflect') if (ph or pw) else arr

    def _sync_crop(self, arrays, mask_arr, cs):
        _, h, w = arrays[0].shape
        top  = random.randint(0, h - cs)
        left = random.randint(0, w - cs)
        return (
            [a[:, top:top + cs, left:left + cs] for a in arrays],
            mask_arr[:, top:top + cs, left:left + cs],
        )

    def _augment(self, arrays, mask_arr):
        if random.random() < 0.5:
            arrays   = [a[:, :, ::-1].copy() for a in arrays]
            mask_arr = mask_arr[:, :, ::-1].copy()
        if random.random() < 0.5:
            arrays   = [a[:, ::-1, :].copy() for a in arrays]
            mask_arr = mask_arr[:, ::-1, :].copy()
        k = random.randint(0, 3)
        if k:
            arrays   = [np.rot90(a, k=k, axes=(1, 2)).copy() for a in arrays]
            mask_arr = np.rot90(mask_arr, k=k, axes=(1, 2)).copy()
        return arrays, mask_arr

    def _resize_and_pad(self, tensor_chw):
        """[C,H,W] → [C, img_size, img_size]，最长边缩放后零填充。"""
        _, h, w = tensor_chw.shape
        scale = self.img_size / max(h, w)
        new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
        resized = F.interpolate(
            tensor_chw.unsqueeze(0).float(),
            size=(new_h, new_w), mode='bilinear', align_corners=False,
        )[0]
        resized = F.pad(resized, (0, self.img_size - new_w, 0, self.img_size - new_h))
        return resized, (new_h, new_w)

    def _build_conversation(self, question, answer):
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], answer)
        return conv.get_prompt()

    def __getitem__(self, idx):
        event = random.choice(self.events)

        s1_pre_path  = random.choice(event['s1_pre'])
        s1_post_path = random.choice(event['s1_post'])

        s1_pre  = _percentile_normalize(_read_bands(s1_pre_path))   # [2, H, W]
        s1_post = _percentile_normalize(_read_bands(s1_post_path))  # [2, H, W]
        mask    = (_read_bands(event['mask'], [1]) > 0).astype(np.float32)  # [1, H, W]

        cs = random.randint(self.min_crop_size, self.crop_size)
        s1_pre  = self._ensure_min_size(s1_pre,  cs)
        s1_post = self._ensure_min_size(s1_post, cs)
        mask    = self._ensure_min_size(mask,    cs)

        (s1_pre, s1_post), mask = self._sync_crop([s1_pre, s1_post], mask, cs)
        (s1_pre, s1_post), mask = self._augment([s1_pre, s1_post], mask)

        # CLIP 输入：伪彩色 RGB（R=VV_pre, G=VV_post, B=VH_pre）
        s1_clip    = np.stack([s1_pre[0], s1_post[0], s1_pre[1]], axis=0)  # [3, H, W]
        s1_clip_u8 = (s1_clip * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        image_clip = self.clip_processor.preprocess(
            s1_clip_u8, return_tensors="pt"
        )["pixel_values"][0]  # [3, 224, 224]

        # 10 通道 tensor：与 WildfireChat 格式完全一致
        #   ch 0-1: S1 pre  (VV, VH)
        #   ch 2-3: S1 post (VV, VH)
        #   ch 4-9: zeros   (无 S2，geo_tower 收到零输入，alignment_loss_weight=0 不参与训练)
        h, w  = s1_pre.shape[1], s1_pre.shape[2]
        zeros6 = np.zeros((6, h, w), dtype=np.float32)
        combined = torch.from_numpy(
            np.concatenate([s1_pre, s1_post, zeros6], axis=0).copy()
        ).float()  # [10, cs, cs]
        images, resize = self._resize_and_pad(combined)  # [10, 1024, 1024]

        # precision cast
        if self.precision == "bf16":
            images     = images.bfloat16()
            image_clip = image_clip.bfloat16()
        elif self.precision == "fp16":
            images     = images.half()
            image_clip = image_clip.half()

        masks = torch.from_numpy(mask[0]).float().unsqueeze(0)  # [1, cs, cs]
        label = torch.ones((cs, cs), dtype=torch.long) * 255

        question     = random.choice(_WILDFIRE_QUESTIONS)
        answer       = random.choice(_WILDFIRE_ANSWERS)
        conversation = self._build_conversation(question, answer)

        return (
            event['id'],    # image_path
            images,         # [3, 1024, 1024]  ← standard SAM input
            image_clip,     # [3, 224, 224]
            [conversation],
            masks,          # [1, cs, cs]
            label,          # [cs, cs]
            resize,         # (h_r, w_r)
            [question],
            ["burn_scar"],
            False,
        )


# ---------------------------------------------------------------------------
# collate_fn  (same structure as utils/dataset.py)
# ---------------------------------------------------------------------------

def collate_fn(batch, tokenizer=None, conv_type="llava_v1",
               use_mm_start_end=True, local_rank=-1):
    (image_path_list, images_list, images_clip_list,
     conversation_list, masks_list, label_list,
     resize_list, questions_list, sampled_classes_list,
     offset_list, inferences) = ([], [], [], [], [], [], [], [], [], [0], [])

    cnt = 0
    for (ip, img, img_clip, convs, masks, label, resize, qs, sc, inf) in batch:
        image_path_list.append(ip)
        images_list.append(img)
        images_clip_list.append(img_clip)
        conversation_list.extend(convs)
        masks_list.append(masks.float())
        label_list.append(label)
        resize_list.append(resize)
        questions_list.append(qs)
        sampled_classes_list.append(sc)
        inferences.append(inf)
        cnt += len(convs)
        offset_list.append(cnt)

    if use_mm_start_end:
        for i in range(len(conversation_list)):
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN,
            )

    input_ids = [
        tokenizer_image_token(p, tokenizer, return_tensors="pt")
        for p in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.conv_templates[conv_type].copy()
    targets = input_ids.clone()
    sep = conv.sep + conv.roles[1] + ": "
    for i, (conversation, target) in enumerate(zip(conversation_list, targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for turn in turns:
            if not turn:
                break
            turn_len = len(tokenizer_image_token(turn, tokenizer))
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += turn_len + 2
        target[cur_len:] = IGNORE_INDEX

    return {
        "image_paths":      image_path_list,
        "images":           torch.stack(images_list),
        "images_clip":      torch.stack(images_clip_list),
        "input_ids":        input_ids,
        "labels":           targets,
        "attention_masks":  attention_masks,
        "masks_list":       masks_list,
        "label_list":       label_list,
        "resize_list":      resize_list,
        "offset":           torch.LongTensor(offset_list),
        "questions_list":   questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inferences":       inferences,
        "conversation_list": conversation_list,
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args(args):
    parser = argparse.ArgumentParser(description="Direct SAR fine-tuning of LISA-7B")
    parser.add_argument("--local_rank",         default=0,    type=int)
    parser.add_argument("--version",            default="/root/autodl-tmp/LISA-7B")
    parser.add_argument("--vision_pretrained",  default="/root/autodl-tmp/sam_vit_h_4b8939.pth")
    parser.add_argument("--vision_tower",       default="openai/clip-vit-large-patch14")
    parser.add_argument("--data_roots",         nargs="+",
                        default=["/root/autodl-tmp/wildfire-dataset-CA-2024"])
    parser.add_argument("--log_base_dir",       default="./runs")
    parser.add_argument("--exp_name",           default="lisa_sar_direct")
    parser.add_argument("--vis_save_path",      default="./vis_output")
    parser.add_argument("--precision",          default="bf16",
                        choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size",         default=1024, type=int)
    parser.add_argument("--model_max_length",   default=512,  type=int)
    parser.add_argument("--epochs",             default=10,   type=int)
    parser.add_argument("--steps_per_epoch",    default=500,  type=int)
    parser.add_argument("--batch_size",         default=1,    type=int)
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--workers",            default=4,    type=int)
    parser.add_argument("--lr",                 default=3e-4, type=float)
    parser.add_argument("--crop_size",          default=512,  type=int)
    parser.add_argument("--min_crop_size",      default=256,  type=int)
    parser.add_argument("--lora_r",             default=8,    type=int)
    parser.add_argument("--lora_alpha",         default=16,   type=int)
    parser.add_argument("--lora_dropout",       default=0.05, type=float)
    parser.add_argument("--lora_target_modules",default="q_proj,v_proj")
    parser.add_argument("--ce_loss_weight",     default=1.0,  type=float)
    parser.add_argument("--dice_loss_weight",   default=0.5,  type=float)
    parser.add_argument("--bce_loss_weight",    default=2.0,  type=float)
    parser.add_argument("--out_dim",            default=256,  type=int)
    parser.add_argument("--beta1",              default=0.9,  type=float)
    parser.add_argument("--beta2",              default=0.95, type=float)
    parser.add_argument("--print_freq",         default=1,    type=int)
    parser.add_argument("--start_epoch",        default=0,    type=int)
    parser.add_argument("--resume",             default="")
    parser.add_argument("--auto_resume",        action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end",   action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--conv_type",          default="llava_v1",
                        choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--gpus",              default="0,1,2,3",
                        help="CUDA_VISIBLE_DEVICES")
    return parser.parse_args(args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    args = parse_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)

    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # ---- Tokenizer ----
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right", use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    if args.use_mm_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.half}.get(args.precision, torch.float32)

    # ---- Model ----
    # WildfireChat 的 LISA.py 有 geo_tower，alignment_loss_weight=0 让它不参与训练
    model_args = {
        "train_mask_decoder":    args.train_mask_decoder,
        "out_dim":               args.out_dim,
        "ce_loss_weight":        args.ce_loss_weight,
        "dice_loss_weight":      args.dice_loss_weight,
        "bce_loss_weight":       args.bce_loss_weight,
        "alignment_loss_weight": 0.0,   # geo_tower 收到零 S2，不计入 loss
        "seg_token_idx":         args.seg_token_idx,
        "vision_pretrained":     args.vision_pretrained,
        "vision_tower":          args.vision_tower,
        "use_mm_start_end":      args.use_mm_start_end,
    }
    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    model.get_model().initialize_lisa_modules(model.get_model().config)

    # Freeze CLIP tower and mm_projector
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # ---- LoRA ----
    if args.lora_r > 0:
        lora_target_modules = [
            n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear)
            and not any(x in n for x in ["visual_model", "vision_tower", "mm_projector", "text_hidden_fcs"])
            and any(x in n for x in args.lora_target_modules.split(","))
        ]
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=sorted(lora_target_modules),
            lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
        ))
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))
    for n, p in model.named_parameters():
        if any(x in n for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]):
            p.requires_grad = True

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # ---- DeepSpeed config ----
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {"type": "AdamW", "params": {
            "lr": args.lr, "weight_decay": 0.0, "betas": [args.beta1, args.beta2],
        }},
        "scheduler": {"type": "WarmupDecayLR", "params": {
            "total_num_steps": args.epochs * args.steps_per_epoch,
            "warmup_min_lr": 0, "warmup_max_lr": args.lr,
            "warmup_num_steps": 100, "warmup_type": "linear",
        }},
        "fp16": {"enabled": args.precision == "fp16"},
        "bf16": {"enabled": args.precision == "bf16"},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True, "overlap_comm": True,
            "reduce_scatter": True, "reduce_bucket_size": 200000000,
            "allgather_partitions": True, "allgather_bucket_size": 200000000,
        },
    }

    # ---- Dataset ----
    world_size = torch.cuda.device_count()
    train_dataset = LISASARDataset(
        data_roots=args.data_roots,
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        samples_per_epoch=(args.batch_size * args.grad_accumulation_steps
                           * args.steps_per_epoch * world_size),
        precision=args.precision,
        image_size=args.image_size,
        crop_size=args.crop_size,
        min_crop_size=args.min_crop_size,
    )
    print(f"Training with {len(train_dataset)} examples.")

    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn, tokenizer=tokenizer,
            conv_type=args.conv_type, use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    if args.auto_resume and not args.resume:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume
    if args.resume:
        model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest")) as f:
            ckpt_dir = f.readline().strip()
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"Resumed from {args.resume}, start epoch {args.start_epoch}")

    train_iter = iter(train_loader)
    for epoch in range(args.start_epoch, args.epochs):
        train_iter = train_one_epoch(
            train_loader, model_engine, epoch, scheduler, writer, train_iter, args
        )
        save_dir = os.path.join(args.log_dir, "ckpt_model")
        if args.local_rank == 0 and os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if args.distributed if hasattr(args, "distributed") else world_size > 1:
            torch.distributed.barrier()
        model_engine.save_checkpoint(save_dir)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(train_loader, model, epoch, scheduler, writer, train_iter, args):
    losses      = AverageMeter("Loss",        ":.4f")
    ce_losses   = AverageMeter("CeLoss",      ":.4f")
    bce_losses  = AverageMeter("MaskBCE",     ":.4f")
    dice_losses = AverageMeter("MaskDICE",    ":.4f")
    mask_losses = AverageMeter("MaskLoss",    ":.4f")
    batch_time  = AverageMeter("Time",        ":6.3f")
    data_time   = AverageMeter("Data",        ":6.3f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [batch_time, losses, ce_losses, mask_losses, bce_losses, dice_losses],
        prefix=f"Epoch: [{epoch}]",
    )
    model.train()
    end = time.time()

    for global_step in range(args.steps_per_epoch):
        for _ in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"]       = input_dict["images"].half()
                input_dict["images_clip"]  = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"]       = input_dict["images"].bfloat16()
                input_dict["images_clip"]  = input_dict["images_clip"].bfloat16()

            output_dict = model(**input_dict)
            loss = output_dict["loss"]
            losses.update(loss.item(),      input_dict["images"].size(0))
            ce_losses.update(output_dict["ce_loss"].item(),        input_dict["images"].size(0))
            bce_losses.update(output_dict["mask_bce_loss"].item(), input_dict["images"].size(0))
            dice_losses.update(output_dict["mask_dice_loss"].item(),input_dict["images"].size(0))
            mask_losses.update(output_dict["mask_loss"].item(),    input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.local_rank == 0:
                progress.display(global_step + 1)
                if writer:
                    writer.add_scalar("train/loss",      losses.avg,      global_step)
                    writer.add_scalar("train/ce_loss",   ce_losses.avg,   global_step)
                    writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                    writer.add_scalar("train/lr",
                                      scheduler.get_last_lr()[0], global_step)
            for m in [batch_time, data_time, losses, ce_losses, bce_losses, dice_losses, mask_losses]:
                m.reset()

    return train_iter


if __name__ == "__main__":
    main(sys.argv[1:])
