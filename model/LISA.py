from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h
from .geo_encoder import GeoEncoder
from .geo_adapter import GeoAdapter
from typing import Optional, List


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        clip_hidden_size = getattr(config, 'mm_hidden_size', 1024)

        self.geo_tower = GeoEncoder(
            output_dim=clip_hidden_size,
            model_name='convnext_tiny',
            freeze_backbone=True
        )

        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.alignment_loss_weight = kwargs.pop("alignment_loss_weight", None)

        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)
        self.geo_adapter = GeoAdapter(dim=1024, depth=3)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pretraining_tp = 1

        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        # Return cached embeddings if available (set by evaluate() to avoid double SAM run)
        if getattr(self, "_sam_embed_cache", None) is not None:
            return self._sam_embed_cache

        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()

                # S1 SAR composite: R=pre_VV(ch0), G=post_VV(ch2), B=pre_VH(ch1)
                # 推理时只有 S1 可用，SAM encoder 与 CLIP Student 保持同源输入
                input_image_rgb = (pixel_values[i, [0, 2, 1], :, :] * 255.0).unsqueeze(0)

                image_embeddings = self.model.visual_model.image_encoder(
                    input_image_rgb
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_clip: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if past_key_values is not None or "past_key_values" in kwargs:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )

        return self.model_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            images_clip=images_clip,
            return_dict=return_dict,
            **kwargs,
        )

    def model_forward(
            self,
            images: torch.FloatTensor,
            images_clip: torch.FloatTensor,
            input_ids: torch.LongTensor,
            labels: Optional[torch.LongTensor] = None,
            attention_mask: torch.LongTensor = None,
            offset: Optional[torch.LongTensor] = None,
            masks_list: Optional[List[torch.FloatTensor]] = None,
            label_list: Optional[List[torch.Tensor]] = None,
            resize_list: Optional[List[tuple]] = None,
            inference: bool = False,
            **kwargs,
    ):
        if attention_mask is None:
            attention_mask = kwargs.get("attention_masks", None)
        attention_masks = attention_mask 

        if offset is None:
            batch_size = input_ids.shape[0]
            offset = torch.arange(batch_size + 1, device=input_ids.device)

        image_embeddings = self.get_visual_embs(images)

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat([seg_token_mask, torch.zeros((seg_token_mask.shape[0], 1)).bool().to(images.device)], dim=1)
        seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], 255)).bool().to(images.device), seg_token_mask], dim=1)

        if not inference:
            images_clip_list = []
            images_geo_list = []
            has_geo_data = images.shape[1] >= 10
            if has_geo_data:
                raw_geo_images = images[:, 4:10, :, :]  # S2 pre(3ch) + post(3ch) = 6ch

            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = images_clip[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous()
                images_clip_list.append(images_clip_i)
                if has_geo_data:
                    images_geo_i = raw_geo_images[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous()
                    images_geo_list.append(images_geo_i)

            images_clip_batched = torch.cat(images_clip_list, dim=0)

            with torch.no_grad():
                clip_feat = self.get_model().get_vision_tower()(images_clip_batched)
                if isinstance(clip_feat, (list, tuple)): clip_feat = clip_feat[0]
                n_tokens = clip_feat.shape[1]
                grid_size = int(n_tokens ** 0.5)

            alignment_loss = torch.tensor(0.0, device=input_ids.device, dtype=clip_feat.dtype)

            if has_geo_data:
                images_geo_batched = torch.cat(images_geo_list, dim=0)
                geo_feat_real = self.model.geo_tower(
                    images_geo_batched,
                    target_size=(grid_size, grid_size)
                )

                if torch.isnan(geo_feat_real).any():
                    geo_feat_real = torch.nan_to_num(geo_feat_real, nan=0.0)

                geo_feat_pred = self.geo_adapter(clip_feat)

                target = geo_feat_real.detach().flatten(0, 1)
                pred = geo_feat_pred.flatten(0, 1)

                target = torch.nan_to_num(target, nan=0.0, posinf=100.0, neginf=-100.0)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=100.0, neginf=-100.0)
                target = torch.clamp(target, min=-100, max=100)
                pred = torch.clamp(pred, min=-100, max=100)

                alignment_loss = F.smooth_l1_loss(pred, target, beta=1.0)

                final_geo = geo_feat_pred

                use_teacher_prob = torch.zeros(1, device=input_ids.device)
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        use_teacher_prob.fill_(torch.rand(1).item())
                    torch.distributed.broadcast(use_teacher_prob, src=0)
                else:
                    use_teacher_prob.fill_(torch.rand(1).item())

                if use_teacher_prob.item() > 1.0:
                    final_geo = geo_feat_real

                fused_feat = clip_feat + final_geo
            else:
                fused_feat = clip_feat
            projected_feat = self.get_model().mm_projector(fused_feat)
            if torch.isnan(projected_feat).any():
                 projected_feat = torch.nan_to_num(projected_feat, nan=0.0)

            _, new_attention_mask, _, new_inputs_embeds, new_labels = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_masks,
                past_key_values=None,
                labels=labels,
                images=None,
                image_features=projected_feat 
            )

            output = super(LlavaLlamaForCausalLM, self).forward(
                inputs_embeds=new_inputs_embeds,
                attention_mask=new_attention_mask,
                labels=new_labels,
                return_dict=True,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        else:
            pass

        if resize_list is None:
            return output

        hidden_states = []
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]

        seg_token_counts = seg_token_mask.int().sum(-1)
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)
        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        pred_masks = []
        for i in range(len(pred_embeddings)):
            (sparse_embeddings, dense_embeddings) = self.model.visual_model.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            
            low_res_masks, _ = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        if inference:
            return {"pred_masks": pred_masks, "gt_masks": masks_list}

        ce_loss = output.loss * self.ce_loss_weight
        
        mask_bce_loss = torch.tensor(0.0, device=ce_loss.device)
        mask_dice_loss = torch.tensor(0.0, device=ce_loss.device)
        num_masks = 0
        
        for batch_idx in range(len(pred_masks)):
            gt_mask = masks_list[batch_idx]
            pred_mask = pred_masks[batch_idx]
            if gt_mask.shape[0] > 0:
                mask_bce_loss += (sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0])
                mask_dice_loss += (dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0])
                num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        if num_masks == 0:
            dummy_loss_sam = 0.0 * sum([p.sum() for p in pred_masks])
            mask_loss = mask_loss + dummy_loss_sam

        alignment_loss = self.alignment_loss_weight * alignment_loss

        loss = ce_loss + mask_loss + alignment_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "align_loss": alignment_loss
        }


    def evaluate(
            self,
            images_clip,
            images,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=32,
            tokenizer=None,
    ):
        with torch.no_grad():
            # ── Step 1: Pre-compute SAM embeddings ONCE and cache them ─────────
            # Caching prevents get_visual_embs() from running SAM again inside
            # model_forward() during generate(), avoiding a costly double SAM run.
            self._sam_embed_cache = self.get_visual_embs(images)
            torch.cuda.empty_cache()

            # ── Step 2: Text generation – no hidden-state accumulation ─────────
            # output_hidden_states=True with max_new_tokens=512 can accumulate
            # >70 GB if KV-cache degrades; generate only the token ids here.
            output_ids = self.generate(
                input_ids=input_ids,
                images=images,
                images_clip=images_clip,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=False,
                return_dict_in_generate=False,
            )
            # output_ids: [batch, full_seq_len]  (input tokens + generated tokens)
            torch.cuda.empty_cache()

            # ── Step 3: Find [SEG] token positions ────────────────────────────
            # output_ids layout: [BOS, img_tok(-200), text..., gen...]
            # After multimodal expansion: [BOS, img_feat×256, text..., gen...]
            # The img_tok at pos 1 expands to 256 tokens → shift downstream by 255.
            # We skip both BOS (pos 0) and img_tok (pos 1), prepend 257 zeros.
            if output_ids.shape[1] <= 2:
                self._sam_embed_cache = None
                return output_ids, []

            seg_from_pos2 = (output_ids[:, 2:] == self.seg_token_idx)  # [B, N+M]
            if not seg_from_pos2.any():
                self._sam_embed_cache = None
                return output_ids, []

            seg_token_mask = torch.cat([
                torch.zeros(seg_from_pos2.shape[0], 257,
                            dtype=torch.bool, device=output_ids.device),
                seg_from_pos2,
            ], dim=1)  # [B, 257+N+M] matches multimodal-expanded sequence length

            # ── Step 4: One targeted LLM forward for SEG-token hidden states ──
            # CLIP features (for building inputs_embeds of the full sequence)
            clip_feat = self.get_model().get_vision_tower()(images_clip[0].unsqueeze(0))
            if isinstance(clip_feat, (list, tuple)):
                clip_feat = clip_feat[0]
            # Inference path: skip geo tower, use CLIP features directly
            projected_feat = self.get_model().mm_projector(clip_feat)
            torch.cuda.empty_cache()

            # Build full-sequence embeddings (replaces img_tok with 256 image features)
            _, full_attn_mask, _, full_embeds, _ = self.prepare_inputs_labels_for_multimodal(
                input_ids=output_ids,
                attention_mask=torch.ones(output_ids.shape, dtype=torch.long,
                                          device=output_ids.device),
                past_key_values=None,
                labels=None,
                images=None,
                image_features=projected_feat,
            )

            # LLM forward – get last-layer hidden states for the full sequence
            lm_out = super(LlavaLlamaForCausalLM, self).forward(
                inputs_embeds=full_embeds,
                attention_mask=full_attn_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_layer_hidden = lm_out.hidden_states[-1]  # [B, full_seq, 4096]
            del lm_out, full_embeds
            torch.cuda.empty_cache()

            # ── Step 5: Extract SEG-token embeddings ──────────────────────────
            assert len(self.model.text_hidden_fcs) == 1
            pred_hidden = self.model.text_hidden_fcs[0](last_layer_hidden)  # [B, seq, out_dim]
            del last_layer_hidden

            pred_embeddings_flat = pred_hidden[seg_token_mask]  # [num_seg, out_dim]
            del pred_hidden

            seg_counts = seg_token_mask.int().sum(-1)
            seg_offset = seg_counts.cumsum(-1)
            seg_offset = torch.cat([
                torch.zeros(1, dtype=torch.long, device=output_ids.device),
                seg_offset,
            ], dim=0)

            pred_embeddings = []
            for i in range(len(seg_offset) - 1):
                s, e = seg_offset[i], seg_offset[i + 1]
                pred_embeddings.append(pred_embeddings_flat[s:e])

            # ── Step 6: SAM mask decoding with pre-computed embeddings ─────────
            image_embeddings = self._sam_embed_cache
            self._sam_embed_cache = None  # release cache

            pred_masks = []
            for i in range(len(pred_embeddings)):
                if pred_embeddings[i].shape[0] == 0:
                    pred_masks.append(torch.zeros(
                        1, *original_size_list[i],
                        device=images.device, dtype=images.dtype))
                    continue
                sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                    points=None, boxes=None, masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, _ = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.get("images", None)
        images_clip = kwargs.get("images_clip", None)
        
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        
        if images is not None:
            _inputs["images"] = images
        if images_clip is not None:
            _inputs["images_clip"] = images_clip
            
        return _inputs
