"""Decoder-only transformer. Attention is written explicitly for inspection."""

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 258
    max_context: int = 128
    width: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    position_embeddings: bool = True

    def __post_init__(self):
        if min(self.vocab_size, self.max_context, self.width, self.heads, self.layers) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.projection = nn.Linear(config.width, config.width, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.max_context, config.max_context, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: Tensor, *, return_weights: bool = False):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.reshape(batch, length, self.heads, self.head_width).transpose(1, 2) for t in (q, k, v)
        )
        # Each query can read its own position and the past, never future targets.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_width)
        scores = scores.masked_fill(~self.causal_mask[:length, :length], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        output = self.dropout(weights) @ v
        output = output.transpose(1, 2).contiguous().reshape(batch, length, width)
        output = self.projection(output)
        return (output, weights) if return_weights else output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = (
            nn.Embedding(config.max_context, config.width) if config.position_embeddings else None
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        # Input and output share a vocabulary representation, counted only once.
        self.output.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: Tensor, targets: Tensor | None = None):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.max_context:
            raise ValueError("Expected [batch, time] tokens within max_context")
        x = self.token_embedding(tokens)
        if self.position_embedding is not None:
            x = x + self.position_embedding(torch.arange(tokens.shape[1], device=tokens.device))
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.output(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        tokens: Tensor,
        *,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 40,
        context_length: int | None = None,
        generator: torch.Generator | None = None,
        eos_id: int = 257,
    ) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] == 0:
            raise ValueError("Generation requires one nonempty prompt")
        if temperature <= 0 or max_new_tokens < 0:
            raise ValueError("temperature must be positive and max_new_tokens nonnegative")
        context = self.config.max_context if context_length is None else context_length
        if not 1 <= context <= self.config.max_context:
            raise ValueError("context_length must be within model capacity")
        if top_k is not None and not 1 <= top_k <= self.config.vocab_size:
            raise ValueError("top_k must be within the vocabulary")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(tokens[:, -context:])
                logits = logits[:, -1, :] / temperature
                logits[:, 256] = float("-inf")  # BOS is input-only.
                if top_k is not None:
                    threshold = torch.topk(logits, top_k).values[:, [-1]]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                next_token = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
                tokens = torch.cat((tokens, next_token), dim=1)
                if next_token.item() == eos_id:
                    break
        finally:
            self.train(was_training)
        return tokens

    @torch.inference_mode()
    def attention_weights(self, tokens: Tensor, layer: int = 0) -> Tensor:
        if self.training:
            raise ValueError("Call eval() before inspecting attention")
        if not 0 <= layer < len(self.blocks):
            raise ValueError("Invalid layer index")
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.max_context:
            raise ValueError("Invalid token shape or context length")
        x = self.token_embedding(tokens)
        if self.position_embedding is not None:
            x = x + self.position_embedding(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks[:layer]:
            x = block(x)
        _, weights = self.blocks[layer].attention(
            self.blocks[layer].attention_norm(x), return_weights=True
        )
        return weights

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
