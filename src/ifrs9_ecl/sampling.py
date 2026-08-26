"""Deterministic bounded sampling for large quarterly archives."""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass
from typing import Iterable, Protocol


class NamedRawRow(Protocol):
    """Minimal row interface required by the sampler."""

    def __getitem__(self, key: str) -> str: ...


@dataclass(frozen=True)
class HashSample:
    """Rows with the smallest stable hash values from one complete source stream."""

    rows: tuple[NamedRawRow, ...]
    population_rows: int
    requested_rows: int
    seed: int

    @property
    def loan_ids(self) -> tuple[str, ...]:
        return tuple(row["loan_id"] for row in self.rows)

    @property
    def loan_id_set(self) -> frozenset[str]:
        return frozenset(self.loan_ids)

    @property
    def sampling_fraction(self) -> float:
        return len(self.rows) / self.population_rows if self.population_rows else 0.0

    @property
    def expansion_weight(self) -> float:
        return self.population_rows / len(self.rows) if self.rows else 0.0


def stable_hash(loan_id: str, seed: int) -> int:
    """Return a platform-independent unsigned 64-bit sampling score."""
    payload = f"{seed}:{loan_id}".encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), byteorder="big"
    )


def select_hash_sample(
    rows: Iterable[NamedRawRow],
    maximum_rows: int,
    *,
    seed: int,
) -> HashSample:
    """Select a deterministic near-uniform sample after scanning the full population."""
    if isinstance(maximum_rows, bool) or maximum_rows < 1:
        raise ValueError("maximum_rows must be a positive integer")
    heap: list[tuple[int, str, NamedRawRow]] = []
    population_rows = 0
    seen_ids: set[str] = set()
    for row in rows:
        population_rows += 1
        loan_id = row["loan_id"]
        if not loan_id:
            raise ValueError(f"Origination row {population_rows} has a blank loan ID")
        if loan_id in seen_ids:
            raise ValueError(f"Duplicate origination loan ID: {loan_id}")
        seen_ids.add(loan_id)
        score = stable_hash(loan_id, seed)
        candidate = (-score, loan_id, row)
        if len(heap) < maximum_rows:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)
    selected = sorted(
        (item[2] for item in heap),
        key=lambda row: row["loan_id"],
    )
    return HashSample(
        rows=tuple(selected),
        population_rows=population_rows,
        requested_rows=maximum_rows,
        seed=seed,
    )

