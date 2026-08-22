"""Frozen-Qwen feature extraction and native K/V evaluation for natural candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time
from typing import Sequence

import numpy as np
import torch

from .features import candidate_features
from .graph import canonical_components
from .natural import NaturalCandidate, NaturalExample


@dataclass(frozen=True)
class CandidateBundle:
    candidates: tuple[NaturalCandidate, ...]
    features: np.ndarray
    token_lengths: tuple[int, ...]
    layer_head_scores: np.ndarray
    adjacency: np.ndarray
    communities: np.ndarray


def compact_candidates(example: NaturalExample, maximum: int = 12) -> tuple[NaturalCandidate, ...]:
    """Bound candidate count while deterministically retaining labeled evidence."""
    if len(example.candidates) <= maximum:
        return example.candidates
    evidence = [value for value in example.candidates if value.is_evidence]
    distractors = [value for value in example.candidates if not value.is_evidence]
    return tuple((evidence + distractors[: max(0, maximum - len(evidence))])[:maximum])


def lexical_score(question: str, text: str) -> float:
    query = {value.strip(".,:;!?()[]{}\"'").lower() for value in question.split()}
    candidate = {value.strip(".,:;!?()[]{}\"'").lower() for value in text.split()}
    query.discard("")
    return len(query & candidate) / max(math.sqrt(len(query) * len(candidate)), 1.0)


class FrozenQwenNaturalAdapter:
    """Small, read-only adapter around a pinned Hugging Face causal LM."""

    def __init__(self, model_id: str, revision: str, device: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        dtype = torch.float16 if self.device.type == "mps" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=dtype, low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()

    def _encode(self, text: str, maximum: int) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)[:maximum]

    def _fixed_chunks(self, example: NaturalExample, maximum: int, chunk_tokens: int = 32):
        chunks = []
        for candidate in example.candidates:
            tokens = self._encode(candidate.text, 100_000)
            for offset in range(0, len(tokens) - chunk_tokens + 1, chunk_tokens):
                piece = tokens[offset : offset + chunk_tokens]
                chunks.append(NaturalCandidate(
                    candidate_id=f"{candidate.candidate_id}:{offset // chunk_tokens}",
                    title=candidate.title, text=self.tokenizer.decode(piece),
                    is_evidence=candidate.is_evidence,
                ))
        if len(chunks) <= maximum:
            return tuple(chunks)
        evidence = [value for value in chunks if value.is_evidence]
        distractors = [value for value in chunks if not value.is_evidence]
        return tuple((evidence + distractors[: max(0, maximum - len(evidence))])[:maximum])

    def _feature_prompt(self, example: NaturalExample, candidates: Sequence[NaturalCandidate]):
        ids: list[int] = []
        spans = []
        lengths = []
        for index, candidate in enumerate(candidates):
            prefix = self._encode(f"\nDocument {index + 1} ({candidate.title}): ", 24)
            content = self._encode(candidate.text, 32)
            ids.extend(prefix)
            start = len(ids)
            ids.extend(content)
            spans.append((start, len(ids)))
            lengths.append(len(prefix) + len(content))
        question = self._encode(f"\nQuestion: {example.question}\nAnswer:", 80)
        ids.extend(question)
        return ids, spans, tuple(lengths)

    @torch.inference_mode()
    def extract(self, example: NaturalExample, *, maximum_candidates: int = 12) -> CandidateBundle:
        candidates = self._fixed_chunks(example, maximum_candidates)
        if len(candidates) < 4 or not any(value.is_evidence for value in candidates):
            raise ValueError(f"{example.example_id} has no usable fixed evidence chunk")
        ids, spans, lengths = self._feature_prompt(example, candidates)
        tensor = torch.tensor([ids], device=self.device)
        output = self.model(tensor, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden = output.hidden_states
        layer_indices = np.linspace(1, len(hidden) - 1, 5, dtype=int)
        groups = 4
        layer_scores = np.zeros((len(layer_indices), groups, len(candidates)), dtype=np.float64)
        adjacency = np.zeros((len(layer_indices), groups, len(candidates), len(candidates)), dtype=np.float64)
        for layer_position, layer_index in enumerate(layer_indices):
            state = hidden[int(layer_index)][0].float()
            question = state[-1]
            vectors = torch.stack([state[start:end].mean(dim=0) for start, end in spans])
            width = vectors.shape[-1] // groups
            for group in range(groups):
                left, right = group * width, (group + 1) * width
                values = torch.nn.functional.normalize(vectors[:, left:right], dim=-1)
                query = torch.nn.functional.normalize(question[left:right], dim=-1)
                layer_scores[layer_position, group] = (values @ query).cpu().numpy()
                adjacency[layer_position, group] = ((values @ values.T + 1.0) / 2.0).cpu().numpy()
        aggregated = adjacency.mean(axis=(0, 1))
        off_diagonal = aggregated[~np.eye(len(candidates), dtype=bool)]
        threshold = float(np.quantile(off_diagonal, 0.65)) if len(off_diagonal) else 0.5
        communities = canonical_components(aggregated, threshold=threshold)
        lexical = np.asarray([lexical_score(example.question, value.text) for value in candidates])
        semantic = layer_scores[-1].mean(axis=0)
        base = 0.45 * lexical + 0.55 * semantic
        features = candidate_features(base, lexical, semantic, layer_scores, adjacency, communities)
        return CandidateBundle(candidates, features, lengths, layer_scores, adjacency, communities)

    @staticmethod
    def _cache_bytes(cache) -> int:
        total = 0
        layers = getattr(cache, "layers", [])
        for layer in layers:
            for tensor in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                if tensor is not None:
                    total += tensor.numel() * tensor.element_size()
        if total:
            return int(total)
        legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
        return int(sum(t.numel() * t.element_size() for pair in legacy for t in pair[:2]))

    @torch.inference_mode()
    def evaluate(self, example: NaturalExample, candidates: Sequence[NaturalCandidate], selected: Sequence[int]) -> dict:
        context_ids = [
            token
            for index in selected
            for token in self._encode(candidates[index].text, 32)
        ]
        tail_prefix = self._encode(f"\nQuestion: {example.question}\nAnswer:", 96)
        answer_ids = self._encode(" " + example.answer, 48)
        if not context_ids or not answer_ids:
            raise ValueError("materialization requires nonempty context and answer")
        if self.device.type == "mps":
            torch.mps.synchronize()
            if hasattr(torch.mps, "reset_peak_memory_stats"):
                torch.mps.reset_peak_memory_stats()
        started = time.perf_counter_ns()
        prefix = self.model(
            torch.tensor([context_ids], device=self.device), use_cache=True, return_dict=True
        )
        if self.device.type == "mps":
            torch.mps.synchronize()
        materialized_ns = time.perf_counter_ns() - started
        cache_bytes = self._cache_bytes(prefix.past_key_values)
        scoring_ids = tail_prefix + answer_ids
        scoring = self.model(
            torch.tensor([scoring_ids], device=self.device),
            past_key_values=prefix.past_key_values, use_cache=False, return_dict=True,
        )
        if self.device.type == "mps":
            torch.mps.synchronize()
        elapsed_ns = time.perf_counter_ns() - started
        logits = scoring.logits[0]
        answer_start = len(tail_prefix)
        positions = torch.arange(answer_start - 1, len(scoring_ids) - 1, device=self.device)
        targets = torch.tensor(answer_ids, device=self.device)
        logprob = torch.log_softmax(logits[positions].float(), dim=-1).gather(1, targets[:, None]).mean()
        greedy = logits[positions].argmax(dim=-1)
        peak = (
            torch.mps.driver_allocated_memory()
            if self.device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory")
            else 0
        )
        return {
            "answer_logprob": float(logprob.cpu()),
            "answer_token_accuracy": float((greedy == targets).float().mean().cpu()),
            "materialized_tokens": len(context_ids),
            "materialized_kv_bytes": cache_bytes,
            "materialization_time_ns": materialized_ns,
            "end_to_end_time_ns": elapsed_ns,
            "peak_device_bytes": int(peak),
            "pid": os.getpid(),
        }
