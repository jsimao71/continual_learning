"""Small inspectable causal Transformer used for controlled experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Intervention:
    layer: int
    component: Literal["sa", "ff"]
    mode: Literal["zero", "mean", "replace", "head_zero", "head_replace"] = "zero"
    replacement: Tensor | None = None
    head: int | None = None


@dataclass
class LayerTrace:
    pre_sa: Tensor
    delta_sa: Tensor
    post_sa: Tensor
    delta_ff: Tensor
    post_block: Tensor
    attention: Tensor
    head_outputs: Tensor | None = None


@dataclass
class ModelTrace:
    layers: list[LayerTrace] = field(default_factory=list)


class InstrumentedBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm_sa = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * mlp_ratio),
            nn.GELU(),
            nn.Linear(width * mlp_ratio, width),
        )

    @staticmethod
    def _intervene(update: Tensor, intervention: Intervention | None) -> Tensor:
        if intervention is None:
            return update
        if intervention.mode == "zero":
            return torch.zeros_like(update)
        if intervention.mode == "mean":
            return update.mean(dim=0, keepdim=True).expand_as(update)
        if intervention.mode == "replace":
            if intervention.replacement is None:
                raise ValueError("replacement intervention requires a tensor")
            return intervention.replacement.to(update).expand_as(update)
        raise ValueError(f"unknown intervention mode: {intervention.mode}")

    def _diagnostic_attention(self, normalized: Tensor, causal_mask: Tensor) -> Tensor:
        """Reconstruct probabilities without changing the model-output kernel."""
        batch, length, width = normalized.shape
        heads = self.attention.num_heads
        head_width = width // heads
        projected = F.linear(normalized, self.attention.in_proj_weight, self.attention.in_proj_bias)
        query, key, _ = projected.chunk(3, dim=-1)
        query = query.view(batch, length, heads, head_width).transpose(1, 2)
        key = key.view(batch, length, heads, head_width).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / (head_width ** 0.5)
        scores = scores + causal_mask[None, None, :, :]
        return torch.softmax(scores, dim=-1)

    def _manual_attention(self, normalized: Tensor, causal_mask: Tensor, zero_head: int | None = None) -> Tensor:
        batch, length, width = normalized.shape
        heads = self.attention.num_heads
        if zero_head is not None and not 0 <= zero_head < heads:
            raise ValueError(f"head index {zero_head} outside [0, {heads})")
        head_width = width // heads
        projected = F.linear(normalized, self.attention.in_proj_weight, self.attention.in_proj_bias)
        query, key, value = projected.chunk(3, dim=-1)
        def split(tensor: Tensor) -> Tensor:
            return tensor.view(batch, length, heads, head_width).transpose(1, 2)
        query, key, value = split(query), split(key), split(value)
        scores = query @ key.transpose(-2, -1) / (head_width ** 0.5)
        probabilities = torch.softmax(scores + causal_mask[None, None, :, :], dim=-1)
        head_outputs = probabilities @ value
        if zero_head is not None:
            head_outputs[:, zero_head] = 0
        joined = head_outputs.transpose(1, 2).reshape(batch, length, width)
        return F.linear(joined, self.attention.out_proj.weight, self.attention.out_proj.bias)

    def _head_contributions(self, normalized: Tensor, causal_mask: Tensor) -> Tensor:
        """Return each head's post-output-projection residual contribution."""
        batch,length,width=normalized.shape;heads=self.attention.num_heads;head_width=width//heads
        projected=F.linear(normalized,self.attention.in_proj_weight,self.attention.in_proj_bias);query,key,value=projected.chunk(3,-1)
        def split(x): return x.view(batch,length,heads,head_width).transpose(1,2)
        query,key,value=split(query),split(key),split(value);probability=torch.softmax(query@key.transpose(-2,-1)/(head_width**.5)+causal_mask[None,None],-1)
        values=probability@value; contributions=[]
        for head in range(heads):
            weight=self.attention.out_proj.weight[:,head*head_width:(head+1)*head_width]
            contributions.append(F.linear(values[:,head],weight))
        return torch.stack(contributions,dim=1)

    def forward(
        self,
        state: Tensor,
        *,
        causal_mask: Tensor,
        capture: bool,
        intervention: Intervention | None,
    ) -> tuple[Tensor, LayerTrace | None]:
        pre_sa = state
        normalized = self.norm_sa(state)
        delta_sa, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
            average_attn_weights=False,
        )
        attention = self._diagnostic_attention(normalized, causal_mask) if capture else None
        head_outputs = self._head_contributions(normalized, causal_mask) if capture or (intervention and intervention.mode == "head_replace") else None
        if intervention and intervention.component == "sa" and intervention.mode == "head_zero":
            if intervention.head is None:
                raise ValueError("head_zero intervention requires a head index")
            manual_full = self._manual_attention(normalized, causal_mask)
            manual_ablated = self._manual_attention(normalized, causal_mask, zero_head=intervention.head)
            effective_sa = delta_sa + (manual_ablated - manual_full)
        elif intervention and intervention.component == "sa" and intervention.mode == "head_replace":
            if intervention.head is None or intervention.replacement is None: raise ValueError("head_replace requires head and replacement")
            effective_sa = delta_sa + intervention.replacement.to(delta_sa).expand_as(delta_sa) - head_outputs[:,intervention.head]
        else:
            effective_sa = self._intervene(delta_sa, intervention if intervention and intervention.component == "sa" else None)
        post_sa = pre_sa + effective_sa
        delta_ff = self.ff(self.norm_ff(post_sa))
        effective_ff = self._intervene(delta_ff, intervention if intervention and intervention.component == "ff" else None)
        post_block = post_sa + effective_ff
        if not capture:
            return post_block, None
        return post_block, LayerTrace(
            pre_sa=pre_sa,
            delta_sa=delta_sa,
            post_sa=post_sa,
            delta_ff=delta_ff,
            post_block=post_block,
            attention=attention,
            head_outputs=head_outputs,
        )


class TinyTransformerLM(nn.Module):
    """A compact LM with exact sublayer trace and intervention locations."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        width: int = 32,
        layers: int = 2,
        heads: int = 2,
        mlp_ratio: int = 2,
        attention_window: int | None = None,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.width = width
        self.attention_window = attention_window
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(max_length, width)
        self.blocks = nn.ModuleList(
            [InstrumentedBlock(width, heads, mlp_ratio) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        *,
        capture: bool = False,
        intervention: Intervention | None = None,
    ) -> tuple[Tensor, ModelTrace | None]:
        batch, length = input_ids.shape
        if length > self.max_length:
            raise ValueError("sequence exceeds configured max_length")
        positions = torch.arange(length, device=input_ids.device)
        state = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        mask = self.causal_mask(length, input_ids.device)
        trace = ModelTrace() if capture else None
        for layer_index, block in enumerate(self.blocks):
            selected = intervention if intervention and intervention.layer == layer_index else None
            state, layer_trace = block(
                state,
                causal_mask=mask,
                capture=capture,
                intervention=selected,
            )
            if trace is not None and layer_trace is not None:
                trace.layers.append(layer_trace)
        return self.lm_head(self.final_norm(state)), trace

    def causal_mask(self, length: int, device: torch.device | str) -> Tensor:
        query = torch.arange(length, device=device)[:, None]
        key = torch.arange(length, device=device)[None, :]
        allowed = key <= query
        if self.attention_window is not None:
            allowed &= key >= query - self.attention_window
        return torch.where(allowed, torch.tensor(0.0, device=device), torch.tensor(float("-inf"), device=device))

    def diagnostic_logits(self, state: Tensor) -> Tensor:
        return self.lm_head(self.final_norm(state))

    @torch.no_grad()
    def target_logprob(
        self,
        input_ids: Tensor,
        target: Tensor,
        intervention: Intervention | None = None,
    ) -> Tensor:
        logits, _ = self(input_ids, intervention=intervention)
        return F.log_softmax(logits[:, -1, :], dim=-1).gather(1, target[:, None]).squeeze(1)


def train_step(model: TinyTransformerLM, batch: Tensor, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(batch[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), batch[:, 1:].reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())
