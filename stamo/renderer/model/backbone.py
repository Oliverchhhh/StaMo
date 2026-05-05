from typing import Any, Dict, List, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from diffusers import SD3Transformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version
from einops import rearrange


class SD3TransformerBackbone(SD3Transformer2DModel):
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
    ):
        super().__init__(
            sample_size,
            patch_size,
            in_channels,
            num_layers,
            attention_head_dim,
            num_attention_heads,
            joint_attention_dim,
            caption_projection_dim,
            pooled_projection_dim,
            out_channels,
            pos_embed_max_size,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        add_hidden_states: torch.FloatTensor = None,
        point_embeds: torch.FloatTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0  # noqa: F841

        # if USE_PEFT_BACKEND:
        #     # weight the lora layers by setting `lora_scale` for each PEFT layer
        #     scale_lora_layers(self, lora_scale)
        # else:
        #     if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
        #         logger.warning(
        #             "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
        #         )

        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        hidden_states_len = hidden_states.shape[1]
        if add_hidden_states is not None:
            add_hidden_states = self.pos_embed(add_hidden_states)
            hidden_states = torch.concat([hidden_states, add_hidden_states], dim=1)

        temb = self.time_text_embed(timestep, pooled_projections)  # timestep
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states[:, :hidden_states_len]

        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                self.out_channels,
                height * patch_size,
                width * patch_size,
            )
        )

        # if USE_PEFT_BACKEND:
        #     # remove `lora_scale` from each PEFT layer
        #     unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class VisionBackbone(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        model_name: str = "vit_large_patch14_dinov2",
        pretrained: bool = True,
        local_ckpt: str = None,
        out_indices: Tuple[int] = (-1,),
    ) -> None:
        super().__init__()
        if local_ckpt:
            cfg = {"file": local_ckpt, "input_size": (3, img_size, img_size)}
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                features_only=True,
                out_indices=out_indices,
                pretrained_cfg_overlay=cfg,
            )
        else:
            cfg = {"input_size": (3, img_size, img_size)}
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
                pretrained_cfg_overlay=cfg,
            )

        assert hasattr(self.model, "feature_info"), (
            "Could not infer vision backbone output channels. Ensure timm version supports features_only=True"
        )

        self.transforms = self.get_transforms()
        self.reduction = self.model.feature_info.reduction()[0]
        self.patches = (img_size // self.reduction) ** 2
        self.channels = self.model.feature_info.channels()[-1]

    def get_transforms(self) -> T.Normalize:
        mean, std = self.model.default_cfg["mean"], self.model.default_cfg["std"]
        transforms = T.Normalize(mean, std)
        return transforms

    def forward(self, images: torch.Tensor):
        features = self.model(images)[-1]
        features = rearrange(features, "b c h w -> b (h w) c")
        return features


class DiTConditionHead(nn.Module):
    def __init__(
        self,
        pooled_dim: int = 2048,
    ):
        super().__init__()
        self.pooled_align_mlp = nn.Linear(4096, pooled_dim)

    def forward(self, inputs: torch.Tensor):
        compress_tokens = inputs  # [bsz, num_token, 2048]
        pooled_embeds = compress_tokens.mean(dim=1)
        pooled_embeds = self.pooled_align_mlp(pooled_embeds)
        return pooled_embeds


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load("./configs/debug.yaml")

    backbone = VisionBackbone(
        img_size=args.data.img_size,
        model_name=args.vision_backbone.model_name,
        pretrained=args.vision_backbone.pretrained,
        local_ckpt=args.vision_backbone.local_ckpt,
    )

    # vit_large_patch14_dinov2                  [bs, 256, 1024]     303.23 M
    # vit_base_patch16_siglip_224.v2_webli      [bs, 196, 768]      85.80 M
    # vit_base_patch16_clip_224.openai          [bs, 196, 768]      85.80 M

    # vit_base_patch14_reg4_dinov2.lvd142m      [bs, 256, 768]      85.73 M
    # vit_base_patch14_dinov2.lvd142m           [bs, 256, 768]      85.72 M
    # vit_so400m_patch14_siglip_224.webli       [bs, 256, 1152]     412.44 M

    images = torch.randn(1, 3, 224, 224)
    features = backbone(images)
    total_params = sum(p.numel() for p in backbone.parameters())

    print("Model name:", args.vision_backbone.model_name)
    print(f"Total params: {total_params:,} ({total_params / 1e6:.2f} M)")
    print("Features shape:", features.shape)
