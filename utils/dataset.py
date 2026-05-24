import glob
import os
import random
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json

from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


class HybridDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        explanatory=0.1,
    ):
        self.samples_per_epoch = samples_per_epoch

        self.height_dataset = HeightDataset(
            base_image_dir=base_image_dir,
            tokenizer=tokenizer,
            vision_tower=vision_tower,
            samples_per_epoch=samples_per_epoch,
            precision=precision,
            image_size=image_size,
            explanatory=explanatory,
        )

    def __len__(self):
        # 返回 HeightDataset 的长度
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # 1. 从 HeightDataset 获取数据
        # LISA 的 Dataset 通常忽略 idx 并在内部随机采样，所以这里调用 self.height_dataset[idx] 即可
        data = self.height_dataset[idx]
        
        # 2. 添加 inference 标志
        # 原 HybridDataset 的逻辑会在最后附加一个 bool 值表示是否为纯推理模式
        inference = False
        
        # 3. 解包数据并返回
        # HeightDataset 返回的是一个 tuple，我们需要用 * 将其解包，再加上 inference
        return *data, inference


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        """
        Modified ValDataset to support HeightBench (RGB+DSM+DTM+SEG).
        Args:
            val_dataset: can be a string path to your validation json folder, 
                         or a specific dataset identifier. 
                         Here we assume it points to 'HeightBench/val' or similar.
        """
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        
        # -----------------------------------------------------------------
        # 1. Load Validation Data (Assume HeightBench format)
        # -----------------------------------------------------------------
        # 我们假设 val_dataset 传入的是类似 "HeightBench/val" 的标识
        # 或者我们硬编码路径去读取 HeightBench 的验证集
        
        # 为了简单起见，这里复用 HeightDataset 的逻辑，读取 validation json
        val_json_dir = os.path.join(base_image_dir, "HeightBench", "val_explanatory") # 假设验证集 JSON 在这里
        # 如果没有专门的文件夹，你可以修改这里指向 train 的一部分，或者一个新的 split
        
        if not os.path.exists(val_json_dir):
             # Fallback: 如果没有 val 文件夹，尝试读取 train 下的 explanatory
             print(f"[WARN] Validation dir {val_json_dir} not found. Fallback to explanatory.")
             val_json_dir = os.path.join(base_image_dir, "HeightBench", "explanatory")
        
        json_files = glob.glob(os.path.join(val_json_dir, "*.json"))
        
        # 如果你想仅验证一部分，可以在这里切片，例如 json_files[:50]
        
        self.img_base_path = os.path.join(base_image_dir, "HeightBench", "train") # 假设图片都在 train 里
        self.data_list = []

        print(f"[INFO] Found {len(json_files)} validation json files.")

        for js in json_files:
            with open(js, "r") as f:
                item = json.load(f)
            
            img_filename = item["image"]
            image_path = os.path.join(self.img_base_path, img_filename)
            
            if os.path.exists(image_path):
                # 存储所有必要信息
                self.data_list.append({
                    "image_path": image_path,
                    "query": item["query"],
                    "outputs": item["outputs"],
                    "mask_name": item.get("mask", ""),
                    "dsm_path": os.path.join(self.img_base_path, item.get("dsm", "")),
                    "dtm_path": os.path.join(self.img_base_path, item.get("dtm", "")),
                    "seg_path": os.path.join(self.img_base_path, item.get("seg", ""))
                })

        print(f"Number of validation samples: {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def preprocess_multimodal(self, rgb: torch.Tensor, dsm: torch.Tensor, dtm: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """
        6-Channel Preprocessing (Same as HeightDataset)
        """
        # 1. RGB
        rgb = (rgb - self.pixel_mean) / self.pixel_std

        # 2. DSM
        MAX_HEIGHT = 500.0 
        dsm = dsm / MAX_HEIGHT
        dsm = torch.clamp(dsm, 0, 1)

        # 3. DTM
        dtm = dtm / MAX_HEIGHT
        dtm = torch.clamp(dtm, 0, 1)

        # 4. SEG
        MAX_CLASSES = 10.0
        seg = seg / MAX_CLASSES

        # 5. Pad
        h, w = rgb.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        
        rgb = F.pad(rgb, (0, padw, 0, padh))
        dsm = F.pad(dsm, (0, padw, 0, padh))
        dtm = F.pad(dtm, (0, padw, 0, padh))
        seg = F.pad(seg, (0, padw, 0, padh))
        
        # 5. Stack
        return torch.cat([rgb, dsm, dtm, seg], dim=0)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_path = item["image_path"]
        
        # 1. Load RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        # 2. Load Geo (DSM & DTM & SEG)
        dsm_path = item["dsm_path"]
        dtm_path = item["dtm_path"]
        seg_path = item["seg_path"]

        if os.path.exists(dsm_path):
            dsm = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)
        else:
            dsm = np.zeros(ori_size, dtype=np.float32)

        if os.path.exists(dtm_path):
            dtm = cv2.imread(dtm_path, cv2.IMREAD_UNCHANGED)
        else:
            dtm = np.zeros(ori_size, dtype=np.float32)

        if os.path.exists(seg_path):
            seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        else:
            seg = np.zeros(ori_size, dtype=np.float32)

        if len(dsm.shape) == 2: dsm = dsm[:, :, None]
        if len(dtm.shape) == 2: dtm = dtm[:, :, None]
        if len(seg.shape) == 2: seg = seg[:, :, None]

        # 3. Load Mask (Ground Truth)
        mask_name = item["mask_name"]
        if mask_name and mask_name != "":
            mask_path = os.path.join(self.img_base_path, mask_name)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, 0)
                mask = (mask > 0).astype(np.uint8) # binary mask
                masks = np.stack([mask], axis=0) # [1, H, W]
            else:
                masks = np.zeros((1, *ori_size), dtype=np.uint8)
        else:
            # 如果没有 mask (text only task)，生成全0
            masks = np.zeros((1, *ori_size), dtype=np.uint8)

        masks = torch.from_numpy(masks)
        # Label is just for compatibility, usually ignored in validation
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        # 4. Prompt Construction
        # Validation 时我们通常固定 prompt 格式
        text = item["query"]
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        
        # 构造问题: <image>\n {query} Please output segmentation mask.
        conv.append_message(
            conv.roles[0],
            DEFAULT_IMAGE_TOKEN
            + "\n {} Please output segmentation mask.".format(text),
        )
        conv.append_message(conv.roles[1], "[SEG].")
        conversations = [conv.get_prompt()]

        # 5. Preprocess Images
        # CLIP
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # SAM & Multimodal
        image_sam = self.transform.apply_image(image)
        dsm_sam = self.transform.apply_image(dsm)
        dtm_sam = self.transform.apply_image(dtm)
        seg_sam = self.transform.apply_image(seg)
        resize = image_sam.shape[:2]

        image_sam_torch = torch.from_numpy(image_sam).permute(2, 0, 1).contiguous().float()
        dsm_sam_torch = torch.from_numpy(dsm_sam).permute(2, 0, 1).contiguous().float()
        dtm_sam_torch = torch.from_numpy(dtm_sam).permute(2, 0, 1).contiguous().float()
        seg_sam_torch = torch.from_numpy(seg_sam).permute(2, 0, 1).contiguous().float()

        final_input_image = self.preprocess_multimodal(image_sam_torch, dsm_sam_torch, dtm_sam_torch, seg_sam_torch)

        inference = True

        return (
            image_path,
            final_input_image,  # [6, 1024, 1024]
            image_clip,         # [3, 224, 224]
            conversations,
            masks,
            labels,
            resize,
            None, # questions (not needed for Val)
            None, # sampled_classes (not needed for Val)
            inference,
        )


