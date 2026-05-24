# Fire-LISA

Fire-LISA is a research codebase for wildfire burn-scar segmentation in
remote-sensing imagery. It adapts LISA with a geospatial alignment branch that
connects SAR-derived CLIP features with Sentinel-2 optical/geospatial teacher
features, then fine-tunes the model to answer segmentation prompts with SAM
mask outputs.

A trained Fire-LISA model is available on Hugging Face:
[`Invisible-dog/Wildfire_Prediction`](https://huggingface.co/Invisible-dog/Wildfire_Prediction).

The project is designed for GPU-based training with DeepSpeed and supports a
two-stage training workflow:

1. Train a `GeoAdapter` to align SAR/CLIP features with geospatial teacher
   features.
2. Fine-tune the Fire-LISA segmentation model with wildfire conversation
   prompts and burn-scar masks.

## Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Architecture](#architecture)
- [Dataset Format](#dataset-format)
- [Installation](#installation)
- [Training](#training)
- [Exporting a Hugging Face Model](#exporting-a-hugging-face-model)
- [Inference](#inference)
- [Alternative SAR-Only Fine-Tuning](#alternative-sar-only-fine-tuning)
- [Script Reference](#script-reference)
- [Monitoring](#monitoring)
- [Implementation Notes](#implementation-notes)
- [License](#license)

## Features

- Two-stage wildfire segmentation training pipeline.
- Sentinel-1 and Sentinel-2 GeoTIFF dataset loaders.
- CLIP-based SAR pseudo-RGB visual input.
- ConvNeXt-based geospatial teacher encoder.
- Trainable GeoAdapter for CLIP-to-geospatial feature alignment.
- Modified LISA model with SAM mask decoding.
- LoRA fine-tuning with DeepSpeed ZeRO stage 2.
- Optional SAR-only baseline training script.

## Repository Structure

```text
.
|-- train_alignment_model.py
|-- train_ds.py
|-- train_ft.py
|-- WildfireChat.py
|-- merge_lora_weights_and_save_hf_model.py
|-- command.txt
|-- requirements.txt
|-- model/
|   |-- LISA.py
|   |-- alignment_model.py
|   |-- geo_encoder.py
|   |-- geo_adapter.py
|   |-- llava/
|   `-- segment_anything/
`-- utils/
    |-- wildfire_dataset.py
    |-- dataset.py
    |-- utils.py
    |-- data_processing.py
    `-- conversation.py
```

Generated outputs such as `runs/`, `alignment_weight*/`, merged model folders,
and Python cache directories are not part of the source tree.

## Architecture

Fire-LISA extends LISA with a geospatial alignment path.

### Input Representations

The main training pipeline uses two image tensors:

| Tensor | Shape | Purpose |
| --- | --- | --- |
| `images` | `[10, 1024, 1024]` | Multi-modal tensor used by SAM and the geospatial branch. |
| `images_clip` | `[3, 224, 224]` | SAR pseudo-RGB tensor processed by CLIP. |

The 10-channel `images` tensor uses the following layout:

| Channels | Source |
| --- | --- |
| `0-1` | Sentinel-1 pre-fire SAR bands: `VV`, `VH` |
| `2-3` | Sentinel-1 post-fire SAR bands: `VV`, `VH` |
| `4-6` | Sentinel-2 pre-fire bands: `B12`, `B8`, `B4` |
| `7-9` | Sentinel-2 post-fire bands: `B12`, `B8`, `B4` |

The CLIP input is a SAR pseudo-RGB composite:

| RGB channel | Source |
| --- | --- |
| `R` | Sentinel-1 pre-fire `VV` |
| `G` | Sentinel-1 post-fire `VV` |
| `B` | Sentinel-1 pre-fire `VH` |

### Training Stages

In stage 1, a frozen CLIP vision tower extracts patch features from the SAR
pseudo-RGB image. A frozen `GeoEncoder` extracts teacher features from the
6-channel Sentinel-2 pre/post tensor. The trainable `GeoAdapter` learns to map
CLIP patch features into the geospatial teacher feature space using Smooth L1
loss.

In stage 2, Fire-LISA loads the trained `GeoAdapter`, freezes it, and fine-tunes
the language and segmentation components. Training samples are formatted as
wildfire segmentation conversations whose answers contain the `[SEG]` token.
The hidden state for `[SEG]` is passed to SAM's prompt encoder and mask decoder
to predict the burn-scar mask.

## Dataset Format

The wildfire loaders expect one or more dataset root directories. Each root
should contain Sentinel-1, Sentinel-2, and mask folders:

```text
wildfire-dataset-CA-2024/
|-- S1_HS/
|   |-- pre/
|   `-- post/
|-- S1_AG/
|   |-- pre/
|   `-- post/
|-- S2_HS/
|   |-- pre/
|   `-- post/
|-- S2_AG/
|   |-- pre/
|   `-- post/
`-- mask/
```

Files are matched by event id. The expected naming pattern is:

```text
CA_<YEAR>_<REGION>_<ID>*.tif
CA_<YEAR>_<REGION>_<ID>_mask.tif
```

Example:

```text
CA_2024_ABC_001_some_source.tif
CA_2024_ABC_001_mask.tif
```

For the main two-stage pipeline, an event is valid only when all of the
following are available:

- Sentinel-1 pre-fire imagery.
- Sentinel-1 post-fire imagery.
- Sentinel-2 pre-fire imagery.
- Sentinel-2 post-fire imagery.
- A binary mask GeoTIFF in `mask/`.

All bands are read with GDAL, converted to `float32`, cleaned of NaN/Inf values,
and normalized per channel with 2nd and 98th percentile clipping.

## Installation

This repository is intended for Linux GPU environments with CUDA.

```bash
pip install -r requirements.txt
```

Additional requirements:

- Install GDAL separately. The code imports `osgeo.gdal`, but GDAL is not listed
  in `requirements.txt`.
- Use a PyTorch, CUDA, and `flash_attn` combination that is compatible with your
  machine.
- Provide local or Hugging Face paths for:
  - Base LISA weights.
  - CLIP vision tower, usually `openai/clip-vit-large-patch14`.
  - SAM ViT-H checkpoint, for example `sam_vit_h_4b8939.pth`.

Common arguments used across scripts:

| Argument | Description |
| --- | --- |
| `--version` | Base LISA model path or Hugging Face model id. |
| `--vision-tower` | CLIP vision tower path or id. |
| `--vision_pretrained` | SAM ViT-H checkpoint path. |
| `--data_roots` | One or more wildfire dataset roots. |
| `--precision` | Training precision: `fp32`, `fp16`, or `bf16`. |

## Training

### Stage 1: GeoAdapter Alignment

Train the `GeoAdapter` before full Fire-LISA fine-tuning:

```bash
python train_alignment_model.py \
  --data_roots /path/to/wildfire-dataset-CA-2022 \
               /path/to/wildfire-dataset-CA-2023 \
               /path/to/wildfire-dataset-CA-2024 \
  --batch_size 10 \
  --epochs 50 \
  --crop_size 256 \
  --save_dir ./alignment_weight_wildfire \
  --clip_model openai/clip-vit-large-patch14 \
  --geo_model convnext_tiny \
  --gpus 0,1
```

Adapter checkpoints are saved after every epoch:

```text
alignment_weight_wildfire/geo_adapter_epoch_<N>.pth
```

Important arguments:

| Argument | Description |
| --- | --- |
| `--data_roots` | Dataset root directories. |
| `--crop_size` | Random crop size for alignment training. |
| `--samples_per_epoch` | Virtual samples per epoch. Samples are randomly drawn from valid events. |
| `--batch_size` | DataLoader batch size. |
| `--epochs` | Number of training epochs. |
| `--lr` | AdamW learning rate for the adapter. |
| `--resume` | Optional adapter checkpoint. |
| `--start_epoch` | Epoch offset for resumed training. |
| `--gpus` | Value assigned to `CUDA_VISIBLE_DEVICES`. |

### Stage 2: Fire-LISA Fine-Tuning

Run `train_ds.py` with DeepSpeed after stage 1:

```bash
deepspeed --num_gpus 2 --master_port 24999 train_ds.py \
  --gpus 0,1 \
  --data_roots /path/to/wildfire-dataset-CA-2024 \
  --version /path/to/LISA-7B \
  --vision-tower /path/to/clip-vit-large-patch14 \
  --vision_pretrained /path/to/sam_vit_h_4b8939.pth \
  --adapter_path ./alignment_weight_wildfire/geo_adapter_epoch_50.pth \
  --log_base_dir ./runs \
  --exp_name wildfire \
  --epochs 5 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --steps_per_epoch 500 \
  --min_crop_size 256 \
  --crop_size 1024 \
  --precision bf16
```

Outputs:

```text
runs/<exp_name>/ckpt_model/
runs/<exp_name>/
```

Important arguments:

| Argument | Description |
| --- | --- |
| `--adapter_path` | Stage-1 `GeoAdapter` checkpoint. |
| `--lora_r` | LoRA rank. Use `0` to disable LoRA. |
| `--lora_target_modules` | Comma-separated target module names, default `q_proj,v_proj`. |
| `--ce_loss_weight` | Language modeling loss weight. |
| `--bce_loss_weight` | Binary mask cross-entropy loss weight. |
| `--dice_loss_weight` | Mask DICE loss weight. |
| `--alignment_loss_weight` | Optional alignment loss weight during stage 2. Defaults to `0.0`. |
| `--auto_resume` | Resume automatically from `runs/<exp_name>/ckpt_model` when present. |
| `--resume` | Explicit DeepSpeed checkpoint directory. |

## Exporting a Hugging Face Model

If training used DeepSpeed ZeRO checkpoints, first convert the checkpoint to a
single PyTorch state dict:

```bash
cd ./runs/wildfire/ckpt_model
python zero_to_fp32.py . ../pytorch_model.bin
```

Then merge LoRA weights and save a Hugging Face-compatible model:

```bash
python merge_lora_weights_and_save_hf_model.py \
  --version /path/to/LISA-7B \
  --weight ./runs/wildfire/pytorch_model.bin \
  --save_path ./fire_lisa_merged \
  --vision_pretrained /path/to/sam_vit_h_4b8939.pth \
  --precision bf16
```

The `--weight` argument can point to:

- A single `.bin` or `.pth` state dict.
- A directory containing sharded Hugging Face checkpoints.
- A `.index.json` file for sharded checkpoints.

The export script merges LoRA adapters with `merge_and_unload()` and saves the
model and tokenizer to `--save_path`.

## Inference

`WildfireChat.py` provides an interactive inference interface for a trained
Fire-LISA model. It loads a local merged model directory or a Hugging Face model
repository, prompts for Sentinel-1 pre-fire and post-fire GeoTIFF paths, accepts
a text query, and writes prediction outputs to disk.

By default, `WildfireChat.py` uses the published model
`Invisible-dog/Wildfire_Prediction`.

Example:

```bash
python WildfireChat.py \
  --version Invisible-dog/Wildfire_Prediction \
  --vis_save_path ./output \
  --precision bf16 \
  --vision_tower openai/clip-vit-large-patch14
```

Because the Hugging Face model is the default, the shorter command is also
valid:

```bash
python WildfireChat.py --vis_save_path ./output --precision bf16
```

Interactive inputs:

```text
S1 pre-fire TIF path (or 'quit'): /path/to/pre_fire.tif
S1 post-fire TIF path: /path/to/post_fire.tif
Prompt: Please segment the wildfire burn scar in this image.
```

The inference script uses Sentinel-1 imagery only. It builds the same SAR
pseudo-RGB representation used during training:

| RGB channel | Source |
| --- | --- |
| `R` | Sentinel-1 pre-fire `VV` |
| `G` | Sentinel-1 post-fire `VV` |
| `B` | Sentinel-1 pre-fire `VH` |

For model compatibility, it creates a 10-channel input where channels `0-3`
contain Sentinel-1 pre/post data and channels `4-9` are zero-filled.

Output files are saved in `--vis_save_path`:

| Output | Description |
| --- | --- |
| `<id>_mask_<n>.jpg` | Predicted binary mask image. |
| `<id>_vis_<n>.jpg` | Mask overlay on the SAR pseudo-RGB visualization. |
| `<id>.json` | Query, prediction text, and input file paths. |

Useful arguments:

| Argument | Description |
| --- | --- |
| `--version` | Local merged model directory or Hugging Face repo id. Defaults to `Invisible-dog/Wildfire_Prediction`. |
| `--vis_save_path` | Directory for masks, overlays, and JSON outputs. |
| `--precision` | `fp32`, `fp16`, or `bf16`. |
| `--vision_tower` | CLIP vision tower path or id. |
| `--load_in_8bit` | Load the language model in 8-bit mode. |
| `--load_in_4bit` | Load the language model in 4-bit mode. |
| `--image_size` | Padded SAM input size, default `1024`. |

## Alternative SAR-Only Fine-Tuning

`train_ft.py` provides a direct SAR-only baseline. It fine-tunes LISA using
Sentinel-1 pre/post imagery and masks without the stage-1 GeoAdapter alignment
workflow.

```bash
deepspeed --include localhost:0,1,2,3 train_ft.py \
  --version /path/to/LISA-7B \
  --vision_pretrained /path/to/sam_vit_h_4b8939.pth \
  --vision_tower openai/clip-vit-large-patch14 \
  --data_roots /path/to/wildfire-dataset-CA-2024 \
  --exp_name lisa_sar_direct \
  --epochs 10 \
  --precision bf16
```

This path keeps the model input shape compatible with Fire-LISA by zero-filling
the geospatial channels and forcing `alignment_loss_weight` to `0.0`.

## Script Reference

| File | Description |
| --- | --- |
| `train_alignment_model.py` | Stage-1 `GeoAdapter` alignment training. |
| `train_ds.py` | Main DeepSpeed fine-tuning script for Fire-LISA. |
| `train_ft.py` | SAR-only direct fine-tuning baseline. |
| `WildfireChat.py` | Interactive inference script for trained Fire-LISA models. |
| `merge_lora_weights_and_save_hf_model.py` | Merges trained weights and exports a Hugging Face model. |
| `utils/wildfire_dataset.py` | Wildfire GeoTIFF datasets for stage 1 and stage 2. |
| `utils/dataset.py` | LISA-compatible collate and validation utilities. |
| `utils/utils.py` | Metrics, progress meters, token constants, and CUDA helpers. |
| `model/LISA.py` | Modified LISA model with geo feature fusion and SAM mask decoding. |
| `model/alignment_model.py` | CLIP, GeoEncoder, and GeoAdapter wrapper for stage 1. |
| `model/geo_encoder.py` | ConvNeXt-based 6-channel geospatial teacher encoder. |
| `model/geo_adapter.py` | Bottleneck residual adapter for CLIP patch tokens. |

## Monitoring

Training log:

```bash
tail -f train.log
```

GPU utilization:

```bash
watch -n 1 nvidia-smi
```

Process check:

```bash
ps aux | grep train_ds.py
```

TensorBoard:

```bash
tensorboard --logdir ./runs
```

## Implementation Notes

- `command.txt` contains example commands from an AutoDL/Linux workflow. Some
  non-English text in that file appears with encoding issues in this checkout,
  but the shell commands remain useful as references.
- `utils/dataset.py` references `HeightDataset`, which is not included in this
  repository. The wildfire training path uses `utils/wildfire_dataset.py`.
- Validation in `train_ds.py` is disabled by default because `--no_eval`
  defaults to `True`.
- `WildfireChat.py` expects CUDA and calls `.cuda()` during preprocessing and
  model loading.
- `model/LISA.py` defines `forward` twice; the second definition is the active
  Python method.
- The training code assumes CUDA availability and uses `.cuda()` in utility
  paths.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for
details.
