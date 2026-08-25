"""Tests for DNA sequence validation and normalisation."""

from __future__ import annotations

import pytest

from evo2_mcp.sequence import (
    LONG_SEQUENCE_WARN_THRESHOLD,
    SequenceValidationError,
    normalise_sequence,
    preview,
)


def test_sequence_validation():
    norm = normalise_sequence("ACGTACGT")
    assert norm.sequence == "ACGTACGT"
    assert norm.length == 8
    assert norm.warning is None


def test_lowercase_sequence():
    assert normalise_sequence("acgtacgt").sequence == "ACGTACGT"
    assert normalise_sequence("aCgT").sequence == "ACGT"


def test_whitespace_handling():
    # spaces, tabs, newlines (FASTA wrapping) are all stripped.
    assert normalise_sequence("AC GT\nAC\tGT").sequence == "ACGTACGT"


def test_invalid_sequence():
    with pytest.raises(SequenceValidationError) as ei:
        normalise_sequence("ACGTXACGT")
    msg = str(ei.value)
    assert "X" in msg
    assert "0-indexed position 4" in msg  # first offending position is pinpointed


def test_invalid_sequence_multiple_bad_chars():
    with pytest.raises(SequenceValidationError):
        normalise_sequence("ACGRYT")


def test_n_rejected_by_default():
    # N is not silently dropped; it is rejected with an explicit error.
    with pytest.raises(SequenceValidationError) as ei:
        normalise_sequence("ACGTNACGT")
    assert "N" in str(ei.value)


def test_n_allowed_when_configured():
    norm = normalise_sequence("ACGTNACGT", allow_ambiguous=True)
    assert norm.sequence == "ACGTNACGT"
    assert "N" in norm.warning  # caveat warning is surfaced


def test_other_iupac_always_rejected():
    for code in "RYSWKMBDHVU":
        with pytest.raises(SequenceValidationError):
            normalise_sequence(f"ACGT{code}T", allow_ambiguous=True)


def test_empty_sequence():
    with pytest.raises(SequenceValidationError):
        normalise_sequence("")
    with pytest.raises(SequenceValidationError):
        normalise_sequence("   \n\t ")


def test_non_string_input():
    with pytest.raises(SequenceValidationError):
        normalise_sequence(12345)


def test_long_sequence_warning():
    seq = "ACGT" * (LONG_SEQUENCE_WARN_THRESHOLD // 4 + 1)
    norm = normalise_sequence(seq)
    assert norm.length == len(seq)
    assert norm.warning is not None


def test_short_sequence_no_warning():
    assert normalise_sequence("ACGT").warning is None


def test_preview_redacts_long_sequences():
    short = preview("ACGT")
    assert short == "ACGT (len=4)"
    long_ = preview("A" * 100)
    assert "..." in long_
    assert "len=100" in long_
