from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from .transcribe import transcribe as transcribe_function
from .decoding import detect_language as detect_language_function, decode as decode_function


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        k_cache: Tensor,
        v_cache: Tensor
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        k = torch.cat([k_cache, k], dim=1)
        v = torch.cat([v_cache, v], dim=1)

        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv), k, v

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(q.shape[0], q.shape[1], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(k.shape[0], k.shape[1], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(v.shape[0], v.shape[1], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        qk = qk + mask[:n_ctx, :n_ctx]

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        k_cache: Tensor,
        v_cache: Tensor
    ):
        q = self.query(x)

        if k_cache.size(1) == 0:
            k = self.key(xa)
            v = self.value(xa)
        else:
            k = k_cache
            v = v_cache

        wv = self.qkv_attention(q, k, v)
        return self.out(wv), k, v

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(q.shape[0], q.shape[1], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(k.shape[0], k.shape[1], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(v.shape[0], v.shape[1], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp),
            nn.GELU(),
            nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
    ):
        dummy_cache = torch.zeros(
            [x.size(0), 0, x.size(-1)], dtype=x.dtype, device=x.device)
        y, _, _ = self.attn(
            self.attn_ln(x),
            mask,
            dummy_cache,
            dummy_cache
        )
        x = x + y
        x = x + self.mlp(self.mlp_ln(x))
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = MultiHeadCrossAttention(n_state, n_head)
        self.cross_attn_ln = nn.LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp),
            nn.GELU(),
            nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        mask: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        xa_k_cache: Tensor,
        xa_v_cache: Tensor
    ):
        y, k_cache, v_cache = self.attn(self.attn_ln(x), mask, k_cache, v_cache)
        x = x + y
        y, xa_k_cache, xa_v_cache = self.cross_attn(self.cross_attn_ln(x), xa, xa_k_cache, xa_v_cache)
        x = x + y
        x = x + self.mlp(self.mlp_ln(x))
        return x, k_cache, v_cache, xa_k_cache, xa_v_cache


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = nn.LayerNorm(n_state)

        mask = torch.zeros(n_ctx, n_ctx)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x, self.mask)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualCrossAttentionBlock] = nn.ModuleList(
            [ResidualCrossAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        xa_k_cache: Tensor,
        xa_v_cache: Tensor
    ):
        offset = k_cache.shape[2]
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)

        k_cache_list, v_cache_list = [], []
        xa_k_cache_list, xa_v_cache_list = [], []
        for i, block in enumerate(self.blocks):
            x, new_k_cache, new_v_cache, new_xa_k_cache, new_xa_v_cache = block(
                x,
                xa,
                self.mask,
                k_cache[:, i, :, :],
                v_cache[:, i, :, :],
                xa_k_cache[:, i, :, :],
                xa_v_cache[:, i, :, :]
            )
            k_cache_list.append(new_k_cache)
            v_cache_list.append(new_v_cache)
            xa_k_cache_list.append(new_xa_k_cache)
            xa_v_cache_list.append(new_xa_v_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return (
            logits,
            torch.stack(k_cache_list, dim=1),
            torch.stack(v_cache_list, dim=1),
            torch.stack(xa_k_cache_list, dim=1),
            torch.stack(xa_v_cache_list, dim=1)
        )


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        dummy_cache = torch.zeros([
            tokens.size(0), self.dims.n_text_layer, 0, self.dims.n_text_state
        ], dtype=audio_features.dtype, device=tokens.device)
        outputs, _, _, _, _ = self.decoder(
            tokens,
            audio_features,
            dummy_cache,
            dummy_cache,
            dummy_cache,
            dummy_cache
        )
        return outputs

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor):
        dummy_cache = torch.zeros([
            tokens.size(0), self.dims.n_text_layer, 0, self.dims.n_text_state
        ], dtype=mel.dtype, device=tokens.device)
        outputs, _, _, _, _ = self.decoder(
            tokens,
            self.encoder(mel),
            dummy_cache,
            dummy_cache,
            dummy_cache,
            dummy_cache
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
