import numpy as np
import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock
from einops import rearrange


class MixAttn(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, head) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=q_dim, num_heads=head, kdim=k_dim, vdim=v_dim, batch_first=True)


class Projector(nn.Module):
    def __init__(self, args, patches: int, channels: int) -> None:
        super().__init__()

        self.patches = patches
        self.channels = channels

        self.hidden_dim: int = args.projector.hidden_dim
        self.cross_attention_dim: int = args.projector.cross_attention_dim
        self.output_align_dim: int = args.projector.output_align_dim

        self.num_token: int = args.projector.num_token
        self.num_attn_layers: int = args.projector.num_attn_layers
        self.num_attn_compress_layers: int = args.projector.num_attn_compress_layers
        self.compress_dims = self._generate_compress_dims()

        self.compress_layers = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(self.compress_dims[:-1], self.compress_dims[1:])],
        )
        self.attn_layers = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.channels,
                    num_attention_heads=8,
                    attention_head_dim=self.channels // 8,
                    dropout=0.1,
                    cross_attention_dim=self.channels,
                )
                for _ in range(self.num_attn_layers)
            ]
        )
        self.attn_compress_layers = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=dim,
                    num_attention_heads=8,
                    attention_head_dim=self.cross_attention_dim // 8,
                    dropout=0.1,
                    cross_attention_dim=self.cross_attention_dim,
                )
                for dim in self.compress_dims
            ]
        )
        self.qkv_layer = nn.Linear(self.channels, self.hidden_dim + self.cross_attention_dim)
        self.compress_align_mlp = nn.Linear(self.patches, self.compress_dims[0])
        self.output_align_mlp = nn.Linear(self.hidden_dim, self.output_align_dim)

    def _generate_compress_dims(self):
        start_exp = self.patches.bit_length() - 1
        end_exp = self.num_token.bit_length() - 1
        exps = np.linspace(start_exp, end_exp, self.num_attn_compress_layers)
        exps = np.ceil(exps).astype(int)
        dims = [2**e for e in exps]
        return dims

    def forward(self, image_embeddings: torch.Tensor):
        hidden_states = image_embeddings.clone()
        for transformer_block in self.attn_layers:
            hidden_states = transformer_block(
                hidden_states=hidden_states,
                encoder_hidden_states=image_embeddings,
            )

        hidden_states = self.qkv_layer(hidden_states)
        q = hidden_states[:, :, : self.hidden_dim]

        encoder_hidden_states = hidden_states[:, :, self.hidden_dim :]
        q = rearrange(q, "b s d -> b d s")
        hidden_states = self.compress_align_mlp(q)

        for compress_block, transformer_block in zip(self.compress_layers, self.attn_compress_layers):
            hidden_states = transformer_block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states = compress_block(hidden_states)

        compressed_embeds = rearrange(hidden_states, "b d s -> b s d")

        # 将特征维度从 2048 对齐到 4096
        compressed_embeds = self.output_align_mlp(compressed_embeds)

        return compressed_embeds


if __name__ == "__main__":
    from backbone import VisionBackbone
    from omegaconf import OmegaConf

    args = OmegaConf.load("./configs/debug.yaml")

    vision_backbone = VisionBackbone(
        model_name=args.vision_backbone.model_name,
        pretrained=args.vision_backbone.pretrained,
        local_ckpt=args.vision_backbone.local_ckpt,
    )

    model = Projector(args, vision_backbone.patches, vision_backbone.channels)

    images = torch.randn((3, 224, 224)).unsqueeze(0)
    image_embeddings = vision_backbone(images)
    compressed_embeds = model(image_embeddings)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Token compression process: {model.compress_dims}")
    print(f"Total params: {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"compressed_embeds.shape: {compressed_embeds.shape}")
