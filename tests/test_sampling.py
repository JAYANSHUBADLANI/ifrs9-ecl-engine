"""Tests for deterministic bounded loan sampling."""

from dataclasses import dataclass

import pytest

from ifrs9_ecl.sampling import select_hash_sample, stable_hash


@dataclass(frozen=True)
class Row:
    loan_id: str

    def __getitem__(self, key: str) -> str:
        if key != "loan_id":
            raise KeyError(key)
        return self.loan_id


def test_hash_sample_is_deterministic_and_file_order_independent() -> None:
    rows = [Row(f"loan-{index}") for index in range(100)]
    first = select_hash_sample(rows, 10, seed=42)
    second = select_hash_sample(reversed(rows), 10, seed=42)
    assert first.loan_ids == second.loan_ids
    assert len(first.rows) == 10
    assert first.population_rows == 100
    assert first.sampling_fraction == 0.1
    assert first.expansion_weight == 10.0


def test_sample_contains_the_smallest_hash_scores() -> None:
    rows = [Row(f"loan-{index}") for index in range(20)]
    sample = select_hash_sample(rows, 5, seed=7)
    expected = sorted(rows, key=lambda row: stable_hash(row.loan_id, 7))[:5]
    assert set(sample.loan_ids) == {row.loan_id for row in expected}


def test_sample_returns_population_when_limit_is_larger() -> None:
    rows = [Row("b"), Row("a")]
    sample = select_hash_sample(rows, 10, seed=1)
    assert sample.loan_ids == ("a", "b")
    assert sample.expansion_weight == 1.0


def test_invalid_limit_and_duplicate_ids_fail() -> None:
    with pytest.raises(ValueError, match="positive"):
        select_hash_sample([], 0, seed=1)
    with pytest.raises(ValueError, match="Duplicate"):
        select_hash_sample([Row("a"), Row("a")], 1, seed=1)

