"""Natural candidate sets for the Paper 1 intervention gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import tarfile
import urllib.request


QASPER_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"


@dataclass(frozen=True)
class NaturalCandidate:
    candidate_id: str
    title: str
    text: str
    is_evidence: bool


@dataclass(frozen=True)
class NaturalExample:
    dataset: str
    example_id: str
    identity_id: str
    split: str
    question: str
    answer: str
    candidates: tuple[NaturalCandidate, ...]


def _partition(identity: str) -> str:
    """Stable approximately even identity-disjoint validation/test split."""
    return "validation" if int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 2 == 0 else "test"


def _balanced_limit(rows: list[NaturalExample], per_split: int, seed: int) -> tuple[NaturalExample, ...]:
    rng = random.Random(seed)
    selected = []
    for split in ("validation", "test"):
        values = [row for row in rows if row.split == split]
        rng.shuffle(values)
        selected.extend(values[:per_split])
    return tuple(sorted(selected, key=lambda row: (row.split, row.example_id)))


def load_hotpotqa(*, per_split: int, seed: int, cache_dir: str | Path) -> tuple[NaturalExample, ...]:
    """Load paragraph candidates with official supporting-fact supervision."""
    from datasets import load_dataset

    source = load_dataset(
        "hotpotqa/hotpot_qa", "distractor", split="validation", cache_dir=str(cache_dir)
    )
    rows = []
    for row in source:
        supporting = {str(title) for title in row["supporting_facts"]["title"]}
        candidates = tuple(
            NaturalCandidate(
                candidate_id=f"{row['id']}:{index}",
                title=str(title),
                text=" ".join(str(value).strip() for value in sentences),
                is_evidence=str(title) in supporting,
            )
            for index, (title, sentences) in enumerate(
                zip(row["context"]["title"], row["context"]["sentences"])
            )
        )
        if len(candidates) >= 4 and any(value.is_evidence for value in candidates):
            identity = str(row["id"])
            rows.append(NaturalExample(
                dataset="hotpotqa", example_id=f"hotpotqa-{identity}", identity_id=identity,
                split=_partition(identity), question=str(row["question"]).strip(),
                answer=str(row["answer"]).strip(), candidates=candidates,
            ))
    return _balanced_limit(rows, per_split, seed)


def _qasper_papers(cache_dir: str | Path) -> dict:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "qasper-train-dev-v0.3.tgz"
    if not archive_path.exists():
        urllib.request.urlretrieve(QASPER_URL, archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.extractfile("qasper-dev-v0.3.json")
        if member is None:
            raise FileNotFoundError("qasper-dev-v0.3.json")
        return json.load(member)


def _qasper_answer(answers: list[dict]) -> tuple[str, list[str]] | None:
    for annotation in answers:
        answer = annotation.get("answer", {})
        evidence = [str(value).strip() for value in answer.get("evidence", []) if str(value).strip()]
        if answer.get("yes_no") is not None:
            return ("yes" if answer["yes_no"] else "no"), evidence
        if str(answer.get("free_form_answer") or "").strip():
            return str(answer["free_form_answer"]).strip(), evidence
        spans = answer.get("extractive_spans") or []
        if spans:
            return str(spans[0]).strip(), evidence
        if answer.get("unanswerable"):
            return "unanswerable", evidence
    return None


def load_qasper(*, per_split: int, seed: int, cache_dir: str | Path) -> tuple[NaturalExample, ...]:
    """Load paragraph candidates; paper identities never cross the split boundary."""
    rows = []
    for paper_id, paper in _qasper_papers(cache_dir).items():
        paragraphs = []
        abstract = str(paper.get("abstract") or "").strip()
        if abstract:
            paragraphs.append(("abstract", abstract))
        for section_index, section in enumerate(paper.get("full_text", [])):
            title = str(section.get("section_name") or f"section-{section_index}")
            paragraphs.extend((title, str(text).strip()) for text in section.get("paragraphs", []) if str(text).strip())
        for qa in paper.get("qas", []):
            resolved = _qasper_answer(qa.get("answers", []))
            if resolved is None:
                continue
            answer, evidence = resolved
            candidates = tuple(
                NaturalCandidate(
                    candidate_id=f"{paper_id}:{index}", title=title, text=text,
                    is_evidence=any(item in text or text in item for item in evidence),
                )
                for index, (title, text) in enumerate(paragraphs)
            )
            if len(candidates) < 4 or not any(value.is_evidence for value in candidates):
                continue
            question_id = str(qa.get("question_id") or len(rows))
            rows.append(NaturalExample(
                dataset="qasper", example_id=f"qasper-{paper_id}-{question_id}",
                identity_id=str(paper_id), split=_partition(str(paper_id)),
                question=str(qa.get("question", "")).strip(), answer=answer,
                candidates=candidates,
            ))
            break  # one scientific unit per paper prevents correlated identities
    return _balanced_limit(rows, per_split, seed)


def identity_disjoint(examples: tuple[NaturalExample, ...]) -> bool:
    validation = {row.identity_id for row in examples if row.split == "validation"}
    test = {row.identity_id for row in examples if row.split == "test"}
    return validation.isdisjoint(test)
