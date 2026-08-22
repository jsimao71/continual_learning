"""Deterministic, provenance-rich experiment artifacts.

The atomic-write and stable-fingerprint design follows the reusable experiment
infrastructure in pdattention/src/common, adapted here to a dependency-free API.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any, length: int | None = None) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


def atomic_write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(dict(row)) + "\n")
    return path


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in materialized))) if materialized else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return path


def _git_value(args: list[str], cwd: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass(frozen=True)
class RunMetadata:
    schema_version: str
    run_id: str
    git_commit: str
    git_dirty: bool
    command: list[str]
    config: dict[str, Any]
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    dataset_id: str
    split: str
    seed: int
    device: str
    dtype: str
    timestamp_utc: str
    python: str
    platform: str
    package_versions: dict[str, str]
    data_hash: str

    @classmethod
    def capture(
        cls,
        *,
        repo: str | Path,
        run_id: str,
        config: dict[str, Any],
        model_id: str,
        dataset_id: str,
        seed: int,
        device: str,
        dtype: str,
        data_hash: str,
        split: str = "synthetic",
        model_revision: str = "local",
        tokenizer_id: str = "integer-tokenizer",
        tokenizer_revision: str = "v1",
    ) -> "RunMetadata":
        import numpy
        import torch

        status = _git_value(["status", "--porcelain"], repo)
        return cls(
            schema_version="cl.run.v1",
            run_id=run_id,
            git_commit=_git_value(["rev-parse", "HEAD"], repo),
            git_dirty=bool(status),
            command=list(sys.argv),
            config=config,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            dataset_id=dataset_id,
            split=split,
            seed=seed,
            device=device,
            dtype=dtype,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            python=sys.version.split()[0],
            platform=platform.platform(),
            package_versions={"numpy": numpy.__version__, "torch": torch.__version__},
            data_hash=data_hash,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

