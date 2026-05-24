"""
WildfireDataset  — Step 1 alignment training
WildfireVLMDataset — Step 2 VLM fine-tuning

每个样本对应一个火灾事件 (CA_YEAR_PROV_ID)。
同一事件下 S1/S2 各有多个 pre 和 post 影像，每次训练随机抽取各一张。

final_input_image : FloatTensor [10, crop_size, crop_size]
    ch 0-1  : S1 pre  (VV, VH)   归一化到 [0,1]
    ch 2-3  : S1 post (VV, VH)
    ch 4-6  : S2 pre  (B12, B8, B4)
    ch 7-9  : S2 post (B12, B8, B4)

image_clip : FloatTensor [3, 224, 224]  ← Student 输入
    S1 SAR 合成 3ch (pre_VV→R, post_VV→G, pre_VH→B)，经 CLIPImageProcessor 处理
Teacher (GeoEncoder in_chans=6) 输入为 final_input_image ch 4:10 (S2 pre+post 6ch)
"""

import os
import re
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from osgeo import gdal
from transformers import CLIPImageProcessor

from model.segment_anything.utils.transforms import ResizeLongestSide
from model.llava import conversation as conversation_lib
from utils.utils import DEFAULT_IMAGE_TOKEN


_FIRE_ID_RE = re.compile(r'^(CA_\d{4}_[A-Z]+_\d+)')
_MASK_RE    = re.compile(r'^(CA_\d{4}_[A-Z]+_\d+)_mask\.tif$')

S1_MODALITIES = ['S1_HS', 'S1_AG']
S2_MODALITIES = ['S2_HS', 'S2_AG']


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _list_by_event(directory):
    """返回 {event_id: [filepath, ...]}，跳过不存在的目录。"""
    result = {}
    if not os.path.isdir(directory):
        return result
    for fname in os.listdir(directory):
        if not fname.lower().endswith('.tif'):
            continue
        m = _FIRE_ID_RE.match(fname)
        if m:
            result.setdefault(m.group(1), []).append(
                os.path.join(directory, fname)
            )
    return result


def _read_bands(path, band_indices=None):
    """
    读取 GeoTIFF，返回 float32 ndarray [C, H, W]。
    band_indices: 1-based list，如 [1,2,3]；None 表示读取全部波段。
    NaN/Inf 替换为 0。
    """
    ds = gdal.Open(path)
    if ds is None:
        raise IOError(f"GDAL cannot open: {path}")
    if band_indices is None:
        band_indices = list(range(1, ds.RasterCount + 1))
    bands = []
    for bi in band_indices:
        arr = ds.GetRasterBand(bi).ReadAsArray().astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        bands.append(arr)
    ds = None
    return np.stack(bands, axis=0)  # [C, H, W]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _percentile_normalize(arr):
    """
    逐通道 percentile 归一化 [p2, p98] → [0, 1]。
    arr: [C, H, W] float32，in-place 修改并返回。
    """
    out = np.empty_like(arr)
    for c in range(arr.shape[0]):
        ch = arr[c]
        valid = ch[np.isfinite(ch)]
        if valid.size == 0:
            out[c] = 0.0
            continue
        p2, p98 = np.percentile(valid, [2, 98])
        rng = p98 - p2
        if rng > 1e-6:
            out[c] = np.clip((ch - p2) / rng, 0.0, 1.0)
        else:
            out[c] = np.zeros_like(ch)
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WildfireDataset(Dataset):
    """
    参数
    ----
    data_roots   : 年份数据集根目录列表，如 [".../2022", ".../2023", ".../2024"]
    vision_tower : CLIP 模型名称或本地路径
    crop_size    : 随机裁剪的正方形边长（像素），默认 256
    samples_per_epoch : 每 epoch 虚拟样本数（实际从事件池中随机采样）
    augment      : 是否启用数据增强（翻转 + 90°旋转）
    """

    def __init__(
        self,
        data_roots,
        vision_tower,
        crop_size=256,
        min_crop_size=128,
        samples_per_epoch=8000,
        augment=True,
    ):
        self.crop_size        = crop_size
        self.min_crop_size    = min_crop_size
        self.samples_per_epoch = samples_per_epoch
        self.augment          = augment
        self.clip_processor   = CLIPImageProcessor.from_pretrained(vision_tower)

        self.events = []  # list of dict，每个 dict 代表一个有效火灾事件

        for root in data_roots:
            # ---- 汇总各模态/pre/post 的文件 ----
            s1_pre  = {}
            s1_post = {}
            s2_pre  = {}
            s2_post = {}

            for mod in S1_MODALITIES:
                for eid, files in _list_by_event(os.path.join(root, mod, 'pre')).items():
                    s1_pre.setdefault(eid, []).extend(files)
                for eid, files in _list_by_event(os.path.join(root, mod, 'post')).items():
                    s1_post.setdefault(eid, []).extend(files)

            for mod in S2_MODALITIES:
                for eid, files in _list_by_event(os.path.join(root, mod, 'pre')).items():
                    s2_pre.setdefault(eid, []).extend(files)
                for eid, files in _list_by_event(os.path.join(root, mod, 'post')).items():
                    s2_post.setdefault(eid, []).extend(files)

            # ---- mask ----
            mask_map = {}
            mask_dir = os.path.join(root, 'mask')
            if os.path.isdir(mask_dir):
                for fname in os.listdir(mask_dir):
                    m = _MASK_RE.match(fname)
                    if m:
                        mask_map[m.group(1)] = os.path.join(mask_dir, fname)

            # ---- 保留四项全有 + mask 的事件 ----
            valid_ids = (
                set(s1_pre) & set(s1_post) &
                set(s2_pre) & set(s2_post) &
                set(mask_map)
            )
            for eid in sorted(valid_ids):
                self.events.append({
                    's1_pre':  s1_pre[eid],
                    's1_post': s1_post[eid],
                    's2_pre':  s2_pre[eid],
                    's2_post': s2_post[eid],
                    'mask':    mask_map[eid],
                    'id':      eid,
                })

        print(f"[WildfireDataset] {len(self.events)} valid fire events "
              f"from {len(data_roots)} year dataset(s)")
        if len(self.events) == 0:
            raise RuntimeError(
                "No valid fire events found. Check data_roots and directory structure."
            )

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------

    def _ensure_min_size(self, arr):
        """如果空间尺寸小于 crop_size，用反射填充补齐。"""
        cs = self.crop_size
        _, h, w = arr.shape
        ph = max(0, cs - h)
        pw = max(0, cs - w)
        if ph > 0 or pw > 0:
            arr = np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode='reflect')
        return arr

    def _random_crop(self, arrays, h, w, cs):
        """对多个 [C,H,W] 数组应用相同的随机裁剪。"""
        top  = random.randint(0, h - cs)
        left = random.randint(0, w - cs)
        return [a[:, top:top + cs, left:left + cs] for a in arrays]

    def _augment(self, arrays):
        """
        对多个 [C,H,W] 数组施加相同的空间增强：
        - 随机水平翻转
        - 随机垂直翻转
        - 随机 90°整数倍旋转
        """
        if random.random() < 0.5:
            arrays = [a[:, :, ::-1].copy() for a in arrays]
        if random.random() < 0.5:
            arrays = [a[:, ::-1, :].copy() for a in arrays]
        k = random.randint(0, 3)
        if k:
            arrays = [np.rot90(a, k=k, axes=(1, 2)).copy() for a in arrays]
        return arrays

    # ------------------------------------------------------------------

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        event = random.choice(self.events)

        # 1. 随机选取各模态的一个影像文件
        s1_pre_path  = random.choice(event['s1_pre'])
        s1_post_path = random.choice(event['s1_post'])
        s2_pre_path  = random.choice(event['s2_pre'])
        s2_post_path = random.choice(event['s2_post'])

        # 2. 读取数据
        #    S1: 全部 2 波段 (VV, VH)
        #    S2: 前 3 波段 (B12, B8, B4)，舍弃第 4 波段 cs_cdf
        s1_pre  = _read_bands(s1_pre_path)            # [2, H, W]
        s1_post = _read_bands(s1_post_path)           # [2, H, W]
        s2_pre  = _read_bands(s2_pre_path,  [1, 2, 3])  # [3, H, W]
        s2_post = _read_bands(s2_post_path, [1, 2, 3])  # [3, H, W]

        # 3. 逐通道 percentile 归一化 → [0, 1]
        s1_pre  = _percentile_normalize(s1_pre)
        s1_post = _percentile_normalize(s1_post)
        s2_pre  = _percentile_normalize(s2_pre)
        s2_post = _percentile_normalize(s2_post)

        # 4. 填充（防止图像比 crop_size 小）
        s1_pre  = self._ensure_min_size(s1_pre)
        s1_post = self._ensure_min_size(s1_post)
        s2_pre  = self._ensure_min_size(s2_pre)
        s2_post = self._ensure_min_size(s2_post)

        # 5. 同步随机裁剪（以 S2 pre 的空间尺寸为基准）
        cs = random.randint(self.min_crop_size, self.crop_size)
        _, h, w = s2_pre.shape
        s1_pre, s1_post, s2_pre, s2_post = self._random_crop(
            [s1_pre, s1_post, s2_pre, s2_post], h, w, cs
        )

        # 6. 数据增强（翻转 + 旋转，四个模态保持一致）
        if self.augment:
            s1_pre, s1_post, s2_pre, s2_post = self._augment(
                [s1_pre, s1_post, s2_pre, s2_post]
            )

        # 7. CLIP 输入（Student）：S1 SAR 合成 3ch
        #    R = S1_pre  VV (ch0)
        #    G = S1_post VV (ch0)
        #    B = S1_pre  VH (ch1)
        #    时序差异合成，使 CLIP 能感知 pre/post 变化
        s1_clip = np.stack([s1_pre[0], s1_post[0], s1_pre[1]], axis=0)  # [3, H, W]
        s1_clip_u8 = (s1_clip * 255).clip(0, 255).astype(np.uint8)
        image_clip = self.clip_processor.preprocess(
            s1_clip_u8.transpose(1, 2, 0), return_tensors="pt"
        )["pixel_values"][0]                                              # [3, 224, 224]

        # 8. 拼接 10 通道输入
        #    ch 0-1 : S1 pre  (VV, VH)
        #    ch 2-3 : S1 post (VV, VH)
        #    ch 4-6 : S2 pre  (B12, B8, B4)
        #    ch 7-9 : S2 post (B12, B8, B4)
        final_input_image = torch.from_numpy(
            np.concatenate([s1_pre, s1_post, s2_pre, s2_post], axis=0).copy()
        ).float()  # [10, crop_size, crop_size]

        # 保持与原 collate_fn 兼容的 9 元素返回格式
        return (
            event['id'],        # [0] 事件 ID（字符串）
            final_input_image,  # [1] [10, H, W]
            image_clip,         # [2] [3, 224, 224]
            None, None, None, None, None, None,
        )


# ---------------------------------------------------------------------------
# Step-2 VLM dataset
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


class WildfireVLMDataset(Dataset):
    """
    Step-2 VLM fine-tuning dataset. 返回与 utils/dataset.py collate_fn 完全兼容
    的 10 元素 tuple:
      (image_path, images, images_clip, conversations, masks,
       label, resize, questions, sampled_classes, inference)

    images [10, 1024, 1024]:
        ch 0-1  : S1 pre  (VV, VH)   [0,1]
        ch 2-3  : S1 post (VV, VH)   [0,1]
        ch 4-6  : S2 pre  (B12,B8,B4) [0,1]
        ch 7-9  : S2 post (B12,B8,B4) [0,1]
    images_clip [3, 224, 224]: S1 SAR 合成 3ch (Student CLIP 输入)
    masks [1, ori_H, ori_W]: 二值化野火烧伤区 mask
    """

    img_size    = 1024
    ignore_label = 255

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
        self.transform         = ResizeLongestSide(image_size)

        # 复用 WildfireDataset 的事件扫描逻辑
        _tmp = WildfireDataset.__new__(WildfireDataset)
        _tmp.crop_size = crop_size
        _tmp.samples_per_epoch = samples_per_epoch
        _tmp.augment = True

        self.events = []
        for root in data_roots:
            s1_pre, s1_post, s2_pre, s2_post = {}, {}, {}, {}
            for mod in S1_MODALITIES:
                for eid, files in _list_by_event(os.path.join(root, mod, 'pre')).items():
                    s1_pre.setdefault(eid, []).extend(files)
                for eid, files in _list_by_event(os.path.join(root, mod, 'post')).items():
                    s1_post.setdefault(eid, []).extend(files)
            for mod in S2_MODALITIES:
                for eid, files in _list_by_event(os.path.join(root, mod, 'pre')).items():
                    s2_pre.setdefault(eid, []).extend(files)
                for eid, files in _list_by_event(os.path.join(root, mod, 'post')).items():
                    s2_post.setdefault(eid, []).extend(files)

            mask_map = {}
            mask_dir = os.path.join(root, 'mask')
            if os.path.isdir(mask_dir):
                for fname in os.listdir(mask_dir):
                    m = _MASK_RE.match(fname)
                    if m:
                        mask_map[m.group(1)] = os.path.join(mask_dir, fname)

            valid_ids = (
                set(s1_pre) & set(s1_post) &
                set(s2_pre) & set(s2_post) & set(mask_map)
            )
            for eid in sorted(valid_ids):
                self.events.append({
                    's1_pre':  s1_pre[eid],
                    's1_post': s1_post[eid],
                    's2_pre':  s2_pre[eid],
                    's2_post': s2_post[eid],
                    'mask':    mask_map[eid],
                    'id':      eid,
                })

        print(f"[WildfireVLMDataset] {len(self.events)} events, "
              f"precision={precision}, crop={crop_size}, img_size={image_size}")

    def __len__(self):
        return self.samples_per_epoch

    # ------------------------------------------------------------------
    # Internal helpers (same as WildfireDataset)
    # ------------------------------------------------------------------

    def _ensure_min_size(self, arr, cs):
        _, h, w = arr.shape
        ph, pw = max(0, cs - h), max(0, cs - w)
        if ph > 0 or pw > 0:
            arr = np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode='reflect')
        return arr

    def _sync_crop(self, arrays, mask_arr, h, w, cs):
        top  = random.randint(0, h - cs)
        left = random.randint(0, w - cs)
        cropped = [a[:, top:top + cs, left:left + cs] for a in arrays]
        cropped_mask = mask_arr[:, top:top + cs, left:left + cs]
        return cropped, cropped_mask

    def _augment(self, arrays, mask_arr):
        if random.random() < 0.5:
            arrays    = [a[:, :, ::-1].copy() for a in arrays]
            mask_arr  = mask_arr[:, :, ::-1].copy()
        if random.random() < 0.5:
            arrays    = [a[:, ::-1, :].copy() for a in arrays]
            mask_arr  = mask_arr[:, ::-1, :].copy()
        k = random.randint(0, 3)
        if k:
            arrays   = [np.rot90(a, k=k, axes=(1, 2)).copy() for a in arrays]
            mask_arr = np.rot90(mask_arr, k=k, axes=(1, 2)).copy()
        return arrays, mask_arr

    def _resize_and_pad(self, tensor_chw):
        """
        [C, H, W] float → [C, img_size, img_size] float
        使用双线性插值 resize 最长边到 img_size，再零填充。
        返回 (resized_tensor, (h_before_pad, w_before_pad))
        """
        _, h, w = tensor_chw.shape
        scale = self.img_size / max(h, w)
        new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
        resized = F.interpolate(
            tensor_chw.unsqueeze(0).float(),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False,
        )[0]
        # 右下零填充
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        resized = F.pad(resized, (0, pad_w, 0, pad_h))
        return resized, (new_h, new_w)

    def _build_conversation(self, question, answer):
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], answer)
        return conv.get_prompt()

    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        event = random.choice(self.events)

        # 1. 随机选文件
        s1_pre_path  = random.choice(event['s1_pre'])
        s1_post_path = random.choice(event['s1_post'])
        s2_pre_path  = random.choice(event['s2_pre'])
        s2_post_path = random.choice(event['s2_post'])

        # 2. 读取
        s1_pre  = _read_bands(s1_pre_path)               # [2, H, W]
        s1_post = _read_bands(s1_post_path)               # [2, H, W]
        s2_pre  = _read_bands(s2_pre_path,  [1, 2, 3])   # [3, H, W]
        s2_post = _read_bands(s2_post_path, [1, 2, 3])   # [3, H, W]
        mask    = _read_bands(event['mask'], [1])         # [1, H, W]
        mask    = (mask > 0).astype(np.float32)

        # 3. 归一化 [0,1]
        s1_pre  = _percentile_normalize(s1_pre)
        s1_post = _percentile_normalize(s1_post)
        s2_pre  = _percentile_normalize(s2_pre)
        s2_post = _percentile_normalize(s2_post)

        # 4. 确保尺寸 ≥ crop_size（用最大值 pad，保证任意随机 cs 都合法）
        max_cs = self.crop_size
        s1_pre  = self._ensure_min_size(s1_pre,  max_cs)
        s1_post = self._ensure_min_size(s1_post, max_cs)
        s2_pre  = self._ensure_min_size(s2_pre,  max_cs)
        s2_post = self._ensure_min_size(s2_post, max_cs)
        mask    = self._ensure_min_size(mask,     max_cs)

        # 5. 随机选 crop size，同步裁剪（以 S2 pre 为基准）
        cs = random.randint(self.min_crop_size, self.crop_size)
        _, h, w = s2_pre.shape
        (s1_pre, s1_post, s2_pre, s2_post), mask = self._sync_crop(
            [s1_pre, s1_post, s2_pre, s2_post], mask, h, w, cs
        )
        ori_size = (cs, cs)   # 裁剪后的原始尺寸，用于 mask gt

        # 6. 数据增强
        (s1_pre, s1_post, s2_pre, s2_post), mask = self._augment(
            [s1_pre, s1_post, s2_pre, s2_post], mask
        )

        # 7. 拼接 10ch tensor，Resize+Pad 到 img_size
        combined = torch.from_numpy(
            np.concatenate([s1_pre, s1_post, s2_pre, s2_post], axis=0).copy()
        ).float()  # [10, cs, cs]
        images, resize = self._resize_and_pad(combined)   # [10, 1024, 1024]

        # 8. CLIP 输入（Student）：S1 SAR 合成 3ch
        #    R = S1_pre VV, G = S1_post VV, B = S1_pre VH
        s1_clip    = np.stack([s1_pre[0], s1_post[0], s1_pre[1]], axis=0)
        s1_clip_u8 = (s1_clip * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        image_clip = self.clip_processor.preprocess(
            s1_clip_u8, return_tensors="pt"
        )["pixel_values"][0]   # [3, 224, 224]

        # 9. Precision 转换
        if self.precision == "bf16":
            images     = images.bfloat16()
            image_clip = image_clip.bfloat16()
        elif self.precision == "fp16":
            images     = images.half()
            image_clip = image_clip.half()

        # 10. Mask tensor（保持原始裁剪尺寸）
        mask_tensor = torch.from_numpy(mask[0]).float()   # [cs, cs]
        masks       = mask_tensor.unsqueeze(0)            # [1, cs, cs]
        label       = torch.ones(ori_size, dtype=torch.long) * self.ignore_label

        # 11. 对话构造
        question = random.choice(_WILDFIRE_QUESTIONS)
        answer   = random.choice(_WILDFIRE_ANSWERS)
        conversation = self._build_conversation(question, answer)

        return (
            event['id'],           # image_path
            images,                # [10, 1024, 1024]
            image_clip,            # [3, 224, 224]
            [conversation],        # conversations
            masks,                 # [1, cs, cs]
            label,                 # [cs, cs]
            resize,                # (h_resized, w_resized)
            [question],            # questions
            ["burn_scar"],         # sampled_classes
            False,                 # inference
        )
