import numpy as np
import pandas as pd
import pytest

from ifrs9_ecl.states import (
    CENSORED_DEFECT,
    CENSORED_RPL,
    CURRENT,
    DEFAULTED,
    DPD_30,
    DPD_60,
    DPD_90_PLUS,
    PAID_OFF,
    UNKNOWN,
    add_state_column,
    coerce_state,
    derive_state,
    derive_state_series,
    map_delinquency_state,
    map_zero_balance_state,
)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("00", CURRENT),
        (0, CURRENT),
        ("01", DPD_30),
        (1.0, DPD_30),
        ("02", DPD_60),
        ("03", DPD_90_PLUS),
        ("04", DPD_90_PLUS),
        (12, DPD_90_PLUS),
        ("03+", DPD_90_PLUS),
        ("RA", DEFAULTED),
        ("ra", DEFAULTED),
        ("XX", UNKNOWN),
        (None, UNKNOWN),
        (np.nan, UNKNOWN),
        ("not-a-code", UNKNOWN),
    ],
)
def test_release_47_delinquency_mapping(code, expected):
    assert map_delinquency_state(code) == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (None, None),
        ("", None),
        ("00", None),
        ("01", PAID_OFF),
        (1, PAID_OFF),
        ("02", DEFAULTED),
        ("03", DEFAULTED),
        ("09", DEFAULTED),
        ("15", DEFAULTED),
        ("16", CENSORED_RPL),
        ("96", CENSORED_DEFECT),
        ("98", CENSORED_DEFECT),
        ("77", UNKNOWN),
    ],
)
def test_release_47_zero_balance_mapping(code, expected):
    assert map_zero_balance_state(code) == expected


def test_zero_balance_terminal_state_overrides_delinquency():
    assert derive_state("02", "01") == PAID_OFF
    assert derive_state("00", "15") == DEFAULTED
    assert derive_state("00", "16") == CENSORED_RPL


def test_missing_zero_balance_code_preserves_delinquency_state():
    assert derive_state("01", None) == DPD_30


def test_unknown_nonmissing_zero_balance_code_is_visible():
    assert derive_state("00", "77") == UNKNOWN


def test_state_aliases_are_normalised_and_bad_labels_raise():
    assert coerce_state("30") == DPD_30
    assert coerce_state("90+") == DPD_90_PLUS
    assert coerce_state(pd.NA) == UNKNOWN
    with pytest.raises(ValueError, match="unsupported loan state"):
        coerce_state("watchlist")


def test_series_derivation_uses_positional_alignment_and_original_index():
    delinquency = pd.Series(["00", "01"], index=[100, 200])
    zero_balance = pd.Series([None, "01"], index=[9, 8])
    result = derive_state_series(delinquency, zero_balance)
    assert result.index.tolist() == [100, 200]
    assert result.tolist() == [CURRENT, PAID_OFF]


def test_series_derivation_rejects_different_lengths():
    with pytest.raises(ValueError, match="equal length"):
        derive_state_series(["00"], [None, "01"])


def test_add_state_column_does_not_mutate_input():
    source = pd.DataFrame({"dq": ["00", "RA"], "zbc": [None, None]})
    result = add_state_column(source, "dq", "zbc")
    assert "state" not in source.columns
    assert result["state"].tolist() == [CURRENT, DEFAULTED]

