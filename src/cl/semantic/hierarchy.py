"""Deterministic balanced hierarchies with exact path and distance metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from cl.common.artifacts import stable_hash


@dataclass(frozen=True)
class HierarchyItem:
    item_id: str
    label: str
    token_id: int
    semantic_family: str
    domain: str
    parent_id: str | None
    depth: int
    natural: bool


class Hierarchy:
    def __init__(self, items: Iterable[HierarchyItem]):
        self.items = {item.item_id: item for item in items}
        if len(self.items) == 0:
            raise ValueError("hierarchy must contain items")
        for item in self.items.values():
            if item.parent_id is not None and item.parent_id not in self.items:
                raise ValueError(f"missing parent {item.parent_id!r}")
            if item.parent_id is not None and self.items[item.parent_id].depth != item.depth - 1:
                raise ValueError("parent depth must be exactly one less than child depth")
        self.version_hash = stable_hash([item.__dict__ for item in sorted(self.items.values(), key=lambda item: item.item_id)])

    def path(self, item_id: str) -> tuple[str, ...]:
        path = []
        current = self.items[item_id]
        while True:
            path.append(current.item_id)
            if current.parent_id is None:
                break
            current = self.items[current.parent_id]
        return tuple(reversed(path))

    def ancestors(self, item_id: str) -> tuple[str, ...]:
        return self.path(item_id)[:-1]

    def siblings(self, item_id: str) -> tuple[str, ...]:
        item = self.items[item_id]
        return tuple(sorted(other.item_id for other in self.items.values() if other.parent_id == item.parent_id and other.item_id != item_id))

    def distance(self, left: str, right: str) -> int:
        left_path, right_path = self.path(left), self.path(right)
        shared = 0
        for a, b in zip(left_path, right_path):
            if a != b:
                break
            shared += 1
        return (len(left_path) - shared) + (len(right_path) - shared)

    def leaves(self, *, family: str | None = None, natural: bool | None = None) -> list[HierarchyItem]:
        parent_ids = {item.parent_id for item in self.items.values() if item.parent_id is not None}
        return sorted(
            [
                item for item in self.items.values()
                if item.item_id not in parent_ids
                and (family is None or item.semantic_family == family)
                and (natural is None or item.natural == natural)
            ],
            key=lambda item: item.item_id,
        )

