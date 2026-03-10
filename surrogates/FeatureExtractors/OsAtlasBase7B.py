import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .Base import BaseFeatureExtractor


class OsAtlasBase7BFeatureExtractor(BaseFeatureExtractor):
    """
    Gray-box feature extractor for OS-Atlas-Base-7B (Qwen2-VL backbone).

    Visual encoder:  Qwen2VisionTransformerPretrainedModel
      - embed_dim  = 1280, num_heads = 16, head_dim = 80, depth = 32
      - patch_size = 14, temporal_patch_size = 2, spatial_merge_size = 2
      - Output dim after PatchMerger: 3584 (LLM hidden_size)

    Text encoder:    LLM word-embedding layer (nn.Embedding, 3584-dim),
                     mean-pooled over tokens → aligned feature space.

    All output tensors follow the shape convention used by EnsembleFeatureLoss:
      [B, N_tokens+1, 3584]  where index-0 is a synthetic "CLS" token (patch mean).
    """

    # Fixed input resolution; must satisfy:
    #   H, W  ≡ 0 (mod patch_size * spatial_merge_size) = 0 (mod 28)
    INPUT_SIZE = 448
    PATCH_SIZE = 14
    TEMPORAL_PATCH_SIZE = 2
    MERGE_SIZE = 2

    MODEL_PATH = "/disk1/users/fengyy/projects/models/OS-Atlas/models/OS-Atlas-Base-7B"

    def __init__(self):
        super().__init__()

        full_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.MODEL_PATH,
            dtype=torch.float32,
            trust_remote_code=True,
        )
        full_model.eval()

        # ── Vision encoder (kept for gradient flow) ──────────────────────────
        self.visual = full_model.visual  # Qwen2VisionTransformerPretrainedModel

        # ── Text embedding layer (LLM word embed, 3584-dim) ──────────────────
        self.embed_tokens = full_model.get_input_embeddings()

        # ── Tokenizer (for tforward) ─────────────────────────────────────────
        self.tokenizer = AutoProcessor.from_pretrained(
            self.MODEL_PATH, trust_remote_code=True
        ).tokenizer

        # Free large LLM layers to save GPU memory
        del full_model.model.language_model.layers
        del full_model.lm_head

        # ── Derived constants ─────────────────────────────────────────────────
        self.grid_h = self.INPUT_SIZE // self.PATCH_SIZE       # 32
        self.grid_w = self.INPUT_SIZE // self.PATCH_SIZE       # 32
        self.grid_t = 1                                        # 1 temporal frame (static image)
        self.n_patches = self.grid_t * self.grid_h * self.grid_w  # 1024
        self.n_merged = self.n_patches // (self.MERGE_SIZE ** 2)  # 256

        # ── Differentiable image normalizer (same mean/std as CLIP/Qwen2VL) ──
        # 使用 Resize((H, W)) 将整张原始图像等比例拉伸（而非先缩最短边再CenterCrop）。
        # 这样整张图像的每个像素都参与特征计算，梯度覆盖全图，
        # 消除"中心有扰动、边缘干净"的裁剪边界伪影。
        self.normalizer = transforms.Compose([
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.Resize(
                (self.INPUT_SIZE, self.INPUT_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    # ─────────────────────────────── helpers ─────────────────────────────────

    def _image_to_flat_patches(self, img: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of normalised images to the flat-patch format expected
        by ``Qwen2VisionTransformerPretrainedModel.patch_embed``.

        This replicates the numpy reshape/transpose in ``Qwen2VLImageProcessor._preprocess``
        in a fully differentiable PyTorch form.

        Args:
            img: [B, C, H, W]  float32, normalised

        Returns:
            flat_patches: [B * grid_t * grid_h * grid_w,
                           C * temporal_patch_size * patch_size * patch_size]
                        = [B * 1024, 1176]

        Patch ordering: every MERGE_SIZE² (=4) consecutive patches are spatially
        adjacent (a 2×2 super-patch), matching what PatchMerger expects.
        """
        B, C, H, W = img.shape
        T = self.TEMPORAL_PATCH_SIZE  # 2
        m = self.MERGE_SIZE            # 2
        gh = H // self.PATCH_SIZE      # 32
        gw = W // self.PATCH_SIZE      # 32
        gt = self.grid_t               # 1
        ph = pw = self.PATCH_SIZE      # 14

        # Duplicate along temporal axis (static image → treat as T identical frames)
        # img_t: [B, T, C, H, W]
        img_t = img.unsqueeze(1).expand(-1, T, -1, -1, -1).contiguous()

        # Decompose spatial dims into (coarse_grid, merge_cell, patch_pixel)
        # [B, gt, T, C, gh//m, m, ph, gw//m, m, pw]
        p = img_t.reshape(B, gt, T, C,
                          gh // m, m, ph,
                          gw // m, m, pw)

        # Permute to: [B, gt, gh//m, gw//m, m_h, m_w, C, T, ph, pw]
        # numpy indices (no batch): (0,3,6,4,7,2,1,5,8)
        # with batch dim prepended:  (0,1,4,7,5,8,3,2,6,9)
        p = p.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).contiguous()

        # Flatten to [B * gt * gh * gw, C * T * ph * pw] = [B*1024, 1176]
        flat = p.reshape(B * gt * gh * gw, C * T * ph * pw)
        return flat

    def _rotary_pos_emb(self, device: torch.device) -> tuple:
        """Rotary position embeddings for the fixed grid (cached-style)."""
        grid_thw = torch.tensor(
            [[self.grid_t, self.grid_h, self.grid_w]], device=device
        )
        rotary = self.visual.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary, rotary), dim=-1)
        return emb.cos(), emb.sin()

    def _cu_seqlens(self, B: int, device: torch.device) -> torch.Tensor:
        """Cumulative sequence lengths for B images at fixed resolution."""
        seq_len = self.grid_t * self.grid_h * self.grid_w  # 1024
        cu = torch.arange(0, (B + 1) * seq_len, seq_len, dtype=torch.int32, device=device)
        return cu

    def _with_cls(self, tokens: torch.Tensor) -> torch.Tensor:
        """Prepend synthetic CLS token (mean of patch tokens) at position 0.

        Args:
            tokens: [B, N, D]
        Returns:
            [B, N+1, D]
        """
        cls = tokens.mean(dim=1, keepdim=True)          # [B, 1, D]
        return torch.cat([cls, tokens], dim=1)           # [B, N+1, D]

    def _run_encoder(
        self,
        flat: torch.Tensor,
        B: int,
        stop_before_last: bool = False,
    ):
        """
        Run patch_embed + visual blocks.

        Returns:
            hidden:     patch hidden states after all (or all-but-last) blocks.
                        Shape [B * n_patches, embed_dim].
            pos_emb:    (cos, sin) rotary position embeddings.
            cu_seqlens: cumulative sequence lengths.
        """
        device = flat.device
        hidden = self.visual.patch_embed(flat)        # [B*N, 1280]

        cos, sin = self._rotary_pos_emb(device)
        position_embeddings = (cos, sin)
        cu_seqlens = self._cu_seqlens(B, device)

        blocks = self.visual.blocks[:-1] if stop_before_last else self.visual.blocks
        for blk in blocks:
            hidden = blk(
                hidden,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
        return hidden, position_embeddings, cu_seqlens

    def _apply_merger(self, hidden: torch.Tensor, B: int) -> torch.Tensor:
        """
        Apply PatchMerger then reshape to [B, n_merged, hidden_size].
        """
        merged = self.visual.merger(hidden)              # [B*n_merged, 3584]
        return merged.reshape(B, self.n_merged, -1)      # [B, 256, 3584]

    # ─────────────────────────── public interface ─────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Global image feature  [B, 3584], L2-normalised.
        Analogous to CLIP's ``get_image_features``.
        """
        img = self.normalizer(x)                        # [B, C, H, W]
        B = img.shape[0]
        flat = self._image_to_flat_patches(img)         # [B*1024, 1176]

        hidden, _, _ = self._run_encoder(flat, B)       # [B*1024, 1280]
        merged = self._apply_merger(hidden, B)          # [B, 256, 3584]

        global_feat = merged.mean(dim=1)                # [B, 3584]
        return F.normalize(global_feat, dim=-1)

    def vforward(
        self,
        x: torch.Tensor,
        enhance: bool = True,
        both: bool = False,
    ) -> torch.Tensor:
        """
        V-type features: value vectors from the last attention block,
        optionally enhanced with V-self-attention, projected through the merger.

        Returns:
            image_embeds:  [B, n_merged+1, 3584]  (token-0 = synthetic CLS)
            (x_feat):      if both=True, also returns xforward output
        """
        img = self.normalizer(x)
        B = img.shape[0]
        flat = self._image_to_flat_patches(img)

        # Run through all blocks except the last one
        layer_input, position_embeddings, cu_seqlens = self._run_encoder(
            flat, B, stop_before_last=True
        )                                               # [B*1024, 1280]

        last_blk = self.visual.blocks[-1]
        num_heads = last_blk.attn.num_heads             # 16
        head_dim = last_blk.attn.head_dim               # 80
        N_total = layer_input.shape[0]                  # B * 1024

        # ── Extract V from last block's fused QKV projection ─────────────────
        norm_x = last_blk.norm1(layer_input)            # [B*N, 1280]
        # qkv: [B*N, 3*1280] → reshape [B*N, 3, heads, head_dim]
        # → permute [3, B*N, heads, head_dim] → unbind
        _, _, v_flat = (
            last_blk.attn.qkv(norm_x)
            .reshape(N_total, 3, num_heads, head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )                                               # v_flat: [B*N, heads, head_dim]

        # ── Enhance: V-self-attention (per image) ────────────────────────────
        N_per_img = self.n_patches                      # 1024
        v = v_flat.reshape(B, N_per_img, num_heads, head_dim)
        v = v.permute(0, 2, 1, 3)                       # [B, heads, N, head_dim]

        if enhance:
            attn_scores = (v @ v.transpose(-2, -1)) / np.sqrt(head_dim)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_out = attn_weights @ v                 # [B, heads, N, head_dim]
        else:
            attn_out = v                                # [B, heads, N, head_dim]

        # [B, heads, N, head_dim] → [B*N, heads*head_dim]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(N_total, num_heads * head_dim)

        # ── Output projection (proj in VisionAttention) ───────────────────────
        proj_v = last_blk.attn.proj(attn_out)           # [B*N, 1280]

        # ── Merge spatial patches ─────────────────────────────────────────────
        merged = self._apply_merger(proj_v, B)          # [B, 256, 3584]

        # ── Prepend synthetic CLS, normalise ─────────────────────────────────
        image_embeds = self._with_cls(merged)           # [B, 257, 3584]
        image_embeds = F.normalize(image_embeds, dim=-1)

        if both:
            x_feat = self.xforward(x)
            return image_embeds, x_feat
        return image_embeds

    def xforward(self, x: torch.Tensor) -> torch.Tensor:
        """
        X-type features: final-layer hidden states after the full encoder,
        projected through the merger.

        Returns:
            [B, n_merged+1, 3584]  (token-0 = synthetic CLS), L2-normalised
        """
        img = self.normalizer(x)
        B = img.shape[0]
        flat = self._image_to_flat_patches(img)

        hidden, _, _ = self._run_encoder(flat, B)       # [B*1024, 1280]
        merged = self._apply_merger(hidden, B)          # [B, 256, 3584]

        x_feat = self._with_cls(merged)                 # [B, 257, 3584]
        return F.normalize(x_feat, dim=-1)

    def tforward(self, text: list) -> torch.Tensor:
        """
        Text features via LLM word-embedding mean-pool  [B, 3584], L2-normalised.

        The word-embedding layer shares the same 3584-dim hidden space as the
        visual merger output, enabling cosine-similarity comparison in the attack loss.

        Args:
            text: list of B strings
        """
        device = next(self.embed_tokens.parameters()).device
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device).unsqueeze(-1).float()

        embeds = self.embed_tokens(input_ids)            # [B, L, 3584]
        # Masked mean pool (ignore padding)
        text_feat = (embeds * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        return F.normalize(text_feat, dim=-1)            # [B, 3584]
