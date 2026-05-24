import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from osgeo import gdal
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from transformers.modeling_utils import load_sharded_checkpoint
from huggingface_hub import snapshot_download

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

import warnings
warnings.filterwarnings("ignore")
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()


# ---------------------------------------------------------------------------
# S1 preprocessing  (与 WildfireDataset 保持一致)
# ---------------------------------------------------------------------------

def _read_bands(path):
    ds = gdal.Open(path)
    if ds is None:
        raise IOError(f"GDAL cannot open: {path}")
    bands = []
    for bi in range(1, ds.RasterCount + 1):
        arr = ds.GetRasterBand(bi).ReadAsArray().astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        bands.append(arr)
    ds = None
    return np.stack(bands, axis=0)  # [C, H, W]


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


def prepare_s1_input(s1_pre_path, s1_post_path, clip_processor, image_size=1024, dtype=torch.bfloat16):
    """
    读取 S1 pre/post GeoTIFF，返回模型所需的两路输入：
      image_clip : [1, 3, 224, 224]  — CLIP / Student 路径
      images     : [1, 10, 1024, 1024] — SAM / Teacher 路径（ch4-9 全零）
    以及可视化用的 SAR 合成图 (H, W, 3) uint8。
    """
    s1_pre  = _percentile_normalize(_read_bands(s1_pre_path))   # [2, H, W]
    s1_post = _percentile_normalize(_read_bands(s1_post_path))  # [2, H, W]

    h, w = s1_pre.shape[1], s1_pre.shape[2]

    # CLIP 输入：S1 SAR 合成 R=pre_VV  G=post_VV  B=pre_VH
    s1_clip_np = np.stack([s1_pre[0], s1_post[0], s1_pre[1]], axis=0)  # [3, H, W]
    s1_clip_hw3 = (s1_clip_np * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    image_clip = (
        clip_processor.preprocess(s1_clip_hw3, return_tensors="pt")["pixel_values"][0]
        .unsqueeze(0).cuda().to(dtype=dtype)
    )

    # 10 通道张量：ch0-1=S1_pre  ch2-3=S1_post  ch4-9=0（推理时无 S2）
    s1_all = np.concatenate([s1_pre, s1_post], axis=0)          # [4, H, W]
    zeros6 = np.zeros((6, h, w), dtype=np.float32)
    img_10ch = torch.from_numpy(np.concatenate([s1_all, zeros6], axis=0))  # [10, H, W]

    # ResizeLongestSide → 零填充到 image_size × image_size
    scale = image_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    img_resized = F.interpolate(img_10ch.unsqueeze(0).float(),
                                size=(new_h, new_w), mode='bilinear', align_corners=False)[0]
    img_padded = F.pad(img_resized, (0, image_size - new_w, 0, image_size - new_h))
    images = img_padded.unsqueeze(0).cuda().to(dtype=dtype)      # [1, 10, 1024, 1024]

    resize_list        = [(new_h, new_w)]
    original_size_list = [(h, w)]

    return image_clip, images, resize_list, original_size_list, s1_clip_hw3


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

DEFAULT_MODEL_REPO = "Invisible-dog/Wildfire_Prediction"


def parse_args(args):
    parser = argparse.ArgumentParser(description="WildfireChat Inference")
    parser.add_argument("--version", default=DEFAULT_MODEL_REPO,
                        help="fine-tuned model dir (local) or Hugging Face repo ID")
    parser.add_argument("--vis_save_path", default="./output", type=str)
    parser.add_argument("--precision", default="bf16", type=str,
                        choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision_tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str,
                        choices=["llava_v1", "llava_llama_2"])
    return parser.parse_args(args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    args = parse_args(args)

    if not os.path.isdir(args.version):
        args.version = snapshot_download(repo_id=args.version)

    os.makedirs(args.vis_save_path, exist_ok=True)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    # ---- dtype / quantization ----
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    if args.load_in_4bit or args.load_in_8bit:
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update({
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model", "vision_tower", "geo_tower", "geo_adapter"],
            ),
        })
    elif args.load_in_8bit:
        kwargs.update({
            "load_in_8bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["visual_model", "vision_tower", "geo_tower", "geo_adapter"],
            ),
        })

    # ---- Model ----
    model = LISAForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        ignore_mismatched_sizes=True,
        **kwargs
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    model.get_model().initialize_lisa_modules(model.get_model().config)

    load_sharded_checkpoint(model, args.version, strict=False)

    if not (args.load_in_4bit or args.load_in_8bit):
        if args.precision == "bf16":
            model = model.bfloat16().cuda()
        elif args.precision == "fp16":
            import deepspeed
            vision_tower_obj = model.get_model().get_vision_tower()
            model.model.vision_tower = None
            model_engine = deepspeed.init_inference(
                model=model, dtype=torch.half,
                replace_with_kernel_inject=True, replace_method="auto",
            )
            model = model_engine.module
            model.model.vision_tower = vision_tower_obj.half().cuda()
        else:
            model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    if hasattr(model.get_model(), "visual_model"):
        model.get_model().visual_model = model.get_model().visual_model.cuda()
    if hasattr(model.get_model(), "geo_tower"):
        model.get_model().geo_tower = model.get_model().geo_tower.cuda()
    if hasattr(model.get_model(), "geo_adapter"):
        model.get_model().geo_adapter = model.get_model().geo_adapter.cuda()

    clip_processor = CLIPImageProcessor.from_pretrained(
        model.config.vision_tower, ignore_mismatched_sizes=True
    )
    model.eval()

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # ---- Interactive loop ----
    while True:
        try:
            s1_pre_path = input("\nS1 pre-fire TIF path (or 'quit'): ").strip()
            if s1_pre_path.lower() in ("quit", "exit", "q"):
                break
            if not os.path.exists(s1_pre_path):
                print("File not found.")
                continue

            s1_post_path = input("S1 post-fire TIF path: ").strip()
            if not os.path.exists(s1_post_path):
                print("File not found.")
                continue

            user_prompt = input("Prompt: ").strip()
            if not user_prompt:
                print("Prompt cannot be empty.")
                continue

            with torch.inference_mode():
                unique_id = os.path.splitext(os.path.basename(s1_pre_path))[0]

                # ---- Build prompt ----
                conv = conversation_lib.conv_templates[args.conv_type].copy()
                conv.messages = []
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + user_prompt
                if args.use_mm_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], "")
                prompt = conv.get_prompt()

                # ---- Preprocess S1 input ----
                image_clip, images, resize_list, original_size_list, vis_np = prepare_s1_input(
                    s1_pre_path, s1_post_path, clip_processor,
                    image_size=args.image_size, dtype=torch_dtype,
                )

                vision_tower_obj = model.get_model().get_vision_tower()
                clip_dtype = (vision_tower_obj.dtype
                              if hasattr(vision_tower_obj, "dtype")
                              else next(vision_tower_obj.parameters()).dtype)
                image_clip = image_clip.to(dtype=clip_dtype)

                input_ids = (
                    tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                    .unsqueeze(0).cuda()
                )

                # ---- Inference ----
                output_ids, pred_masks = model.evaluate(
                    image_clip,
                    images,
                    input_ids,
                    resize_list,
                    original_size_list,
                    max_new_tokens=512,
                    tokenizer=tokenizer,
                )

                output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
                text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
                if "ASSISTANT:" in text_output:
                    text_output = text_output.split("ASSISTANT: ")[-1]
                text_output = text_output.replace("\n", "").replace("  ", " ").strip()
                print(f"Prediction: {text_output}")

                # ---- Save masks + visualization ----
                for i, pred_mask in enumerate(pred_masks):
                    if pred_mask.shape[0] == 0:
                        continue
                    pred_mask = pred_mask.detach().cpu().numpy()[0] > 0

                    save_base = os.path.join(args.vis_save_path, f"{unique_id}_mask_{i}")
                    cv2.imwrite(f"{save_base}.jpg", pred_mask.astype(np.uint8) * 100)

                    # 叠加在 SAR 合成图上（BGR for cv2）
                    vis_bgr = cv2.cvtColor(vis_np, cv2.COLOR_RGB2BGR)
                    overlay = vis_bgr.copy()
                    overlay[pred_mask] = (
                        vis_bgr * 0.5
                        + pred_mask[:, :, None].astype(np.uint8) * np.array([0, 0, 255]) * 0.5
                    )[pred_mask]
                    cv2.imwrite(
                        os.path.join(args.vis_save_path, f"{unique_id}_vis_{i}.jpg"),
                        overlay,
                    )

                result_item = {
                    "unique_id":   unique_id,
                    "s1_pre":      s1_pre_path,
                    "s1_post":     s1_post_path,
                    "query":       user_prompt,
                    "prediction":  text_output,
                }
                with open(os.path.join(args.vis_save_path, f"{unique_id}.json"), "w") as f:
                    json.dump(result_item, f, indent=4, ensure_ascii=False)

                torch.cuda.empty_cache()

        except KeyboardInterrupt:
            break
        except Exception as e:
            import traceback
            print(f"Error: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main(sys.argv[1:])
