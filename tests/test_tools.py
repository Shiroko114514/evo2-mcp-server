"""Tool-level tests: evo2_forward, evo2_score, evo2_variant_score,
evo2_batch_score, evo2_score_fasta. All API calls go through FakeEvo2Client."""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest

from evo2_mcp.api_client import Evo2APIError
from evo2_mcp.fasta import FastaError
from evo2_mcp.forward_output import _log_softmax
from evo2_mcp.sequence import SequenceValidationError
from evo2_mcp.tools import (
    evo2_batch_score,
    evo2_forward,
    evo2_score,
    evo2_score_fasta,
    evo2_variant_score,
)

from conftest import FakeEvo2Client, NUC_INDEX, favor_logits, uniform_logits

# 20 bp: A(0) C(1) G(2) T(3) | A(4) C(5) G(6) T(7) | A(8) C(9) G(10) T(11) |
#        A(12) C(13) G(14) T(15) | A(16) C(17) G(18) T(19)
SEQ = "ACGTACGTACGTACGTACGT"
LOG_512 = math.log(512.0)


def expected_total(seq: str, logits: np.ndarray) -> float:
    """Reference total: log_softmax(logits[:-1]) gathered at seq[1:] bytes."""
    logp = _log_softmax(logits[:-1], axis=-1)
    target = np.array([NUC_INDEX[c] for c in seq[1:]])
    return float(logp[np.arange(len(target)), target].sum())


def next_base_logits(seq: str, logit: float = 2.0) -> np.ndarray:
    """Logits that favour each position's actual next base.

    logits[i, seq[i+1]] = logit, everything else 0 -> every scored position
    has logp = logit - logsumexp(logit, 0*511); a variant at 1-based position
    p changes exactly one scored prediction, so delta = -logit exactly.
    """
    arr = np.zeros((len(seq), 512), dtype=np.float32)
    for i in range(len(seq) - 1):
        arr[i, NUC_INDEX[seq[i + 1]]] = logit
    return arr


# ---------------------------------------------------------------------------
# evo2_forward
# ---------------------------------------------------------------------------

def test_forward_summary(settings):
    client = FakeEvo2Client({"output_layer": uniform_logits(len(SEQ)).reshape(len(SEQ), 1, 512)})
    result = asyncio.run(evo2_forward(sequence=SEQ, settings=settings, client=client))
    assert result["sequence_length"] == 20
    assert result["returned_layers"] == ["output_layer"]
    stats = result["layer_stats"][0]
    assert stats["shape"] == [20, 1, 512]
    assert stats["dtype"] == "float32"
    assert "arrays" not in result
    assert result["warnings"] == []


def test_forward_normalises_input(settings):
    client = FakeEvo2Client({"output_layer": uniform_logits(6)})
    asyncio.run(evo2_forward(sequence="acgt acgt\nacgt", settings=settings, client=client))
    assert client.calls[0][0] == "ACGTACGTACGT"
    assert client.calls[0][1] == ["output_layer"]


def test_forward_save_mode(settings):
    client = FakeEvo2Client({"output_layer": uniform_logits(6)})
    result = asyncio.run(evo2_forward(sequence=SEQ, mode="save", settings=settings, client=client))
    assert result["saved"] is True
    assert result["path"].startswith(str(settings.output_dir))
    assert result["bytes_on_disk"] > 0


def test_forward_save_path_sandboxed(settings):
    client = FakeEvo2Client({"output_layer": uniform_logits(6)})
    outside = settings.output_dir.parent / "elsewhere" / "x.npz"
    with pytest.raises(ValueError, match="output directory"):
        asyncio.run(evo2_forward(
            sequence=SEQ, save_path=str(outside), settings=settings, client=client
        ))


def test_forward_save_path_inside_output_dir(settings):
    client = FakeEvo2Client({"output_layer": uniform_logits(6)})
    target = settings.output_dir / "sub" / "custom.npz"
    result = asyncio.run(evo2_forward(
        sequence=SEQ, save_path=str(target), settings=settings, client=client
    ))
    assert result["path"] == str(target.resolve())


def test_forward_raw_small(settings):
    arr = uniform_logits(4)  # 2048 values <= 4096 cap
    client = FakeEvo2Client({"output_layer": arr})
    result = asyncio.run(evo2_forward(sequence="ACGTACGT", mode="raw", settings=settings, client=client))
    assert "arrays" in result
    assert len(result["arrays"]["output_layer"]) == 4


def test_forward_raw_too_large(settings):
    arr = uniform_logits(100)  # 51200 values > 4096 cap
    client = FakeEvo2Client({"output_layer": arr})
    with pytest.raises(ValueError, match="Refusing to inline"):
        asyncio.run(evo2_forward(sequence="ACGT" * 25, mode="raw", settings=settings, client=client))


def test_forward_invalid_sequence(settings):
    client = FakeEvo2Client({"output_layer": uniform_logits(4)})
    with pytest.raises(SequenceValidationError):
        asyncio.run(evo2_forward(sequence="ACGTX", settings=settings, client=client))
    assert client.calls == []  # no API call for invalid input


# ---------------------------------------------------------------------------
# evo2_score
# ---------------------------------------------------------------------------

def test_score_total_and_mean(settings):
    logits = uniform_logits(len(SEQ))
    client = FakeEvo2Client({"output_layer": logits})
    result = asyncio.run(evo2_score(sequence=SEQ, settings=settings, client=client))
    assert result["likelihood_computed"] is True
    assert result["scored_positions"] == 19
    assert result["total_log_likelihood"] == pytest.approx(19 * -LOG_512)
    assert result["mean_log_likelihood"] == pytest.approx(-LOG_512)
    assert result["per_position_log_likelihood"] is None


def test_score_per_position(settings):
    logits = next_base_logits(SEQ)
    client = FakeEvo2Client({"output_layer": logits})
    result = asyncio.run(evo2_score(
        sequence=SEQ, include_per_position=True, settings=settings, client=client
    ))
    assert len(result["per_position_log_likelihood"]) == 19
    assert result["per_position_truncated"] is False
    assert result["total_log_likelihood"] == pytest.approx(expected_total(SEQ, logits))
    assert result["total_log_likelihood"] == pytest.approx(19 * (2.0 - math.log(math.exp(2.0) + 511)))


def test_score_per_position_truncated(settings):
    seq = "ACGT" * 3000  # 12000 bp -> 11999 values > max_per_position (5000)
    logits = uniform_logits(len(seq))
    client = FakeEvo2Client({"output_layer": logits})
    result = asyncio.run(evo2_score(
        sequence=seq, include_per_position=True, settings=settings, client=client
    ))
    assert result["per_position_truncated"] is True
    assert result["per_position_total"] == 11999
    assert len(result["per_position_log_likelihood"]) == settings.max_per_position


def test_score_misaligned_falls_back_honestly(settings):
    # logits seq_len (30) != sequence length (20): must NOT fabricate a score.
    client = FakeEvo2Client({"output_layer": uniform_logits(30)})
    result = asyncio.run(evo2_score(sequence=SEQ, settings=settings, client=client))
    assert result["likelihood_computed"] is False
    assert "not supported" in result["message"].lower()
    assert "layer_stats" in result


def test_score_too_short(settings):
    client = FakeEvo2Client({"output_layer": uniform_logits(1)})
    with pytest.raises(SequenceValidationError, match="at least 2"):
        asyncio.run(evo2_score(sequence="A", settings=settings, client=client))


# ---------------------------------------------------------------------------
# evo2_variant_score
# ---------------------------------------------------------------------------

def test_variant_validation(settings):
    client = FakeEvo2Client({"output_layer": uniform_logits(len(SEQ))})
    cases = [
        dict(sequence=SEQ, position=0, ref="A", alt="G"),   # 1-based position 0 invalid
        dict(sequence=SEQ, position=21, ref="A", alt="G"),  # out of range
        dict(sequence=SEQ, position=3, ref="C", alt="G"),   # ref mismatch (index 2 = 'G')
        dict(sequence=SEQ, position=2, ref="A", alt="A"),   # ref == alt (also a mismatch)
        dict(sequence=SEQ, position=2, ref="C", alt="NN"),  # bad alt
        dict(sequence=SEQ, position=2, ref="C", alt="N"),   # N allele rejected
        dict(sequence=SEQ, position=1, ref="A", alt="G"),   # first position unscorable
    ]
    for kwargs in cases:
        with pytest.raises((ValueError, SequenceValidationError)):
            asyncio.run(evo2_variant_score(**kwargs, settings=settings, client=client))
    assert client.calls == []  # nothing reached the API


def test_variant_score(settings):
    """Required test name: variant scoring end-to-end (mock API)."""
    logits = next_base_logits(SEQ)
    client = FakeEvo2Client({"output_layer": logits})
    result = asyncio.run(evo2_variant_score(
        sequence=SEQ, position=10, ref="C", alt="G", settings=settings, client=client
    ))
    assert result["delta_log_likelihood"] == pytest.approx(-2.0)
    assert result["wildtype_log_likelihood"] == pytest.approx(expected_total(SEQ, logits))


def test_variant_score_math(settings):
    # next-base-favouring logits -> a C->G variant (index 9) flips exactly one
    # scored prediction, so delta_log_likelihood == -logit == -2.0 exactly.
    logits = next_base_logits(SEQ, logit=2.0)
    client = FakeEvo2Client({"output_layer": logits})

    result = asyncio.run(evo2_variant_score(
        sequence=SEQ, position=10, ref="C", alt="G", settings=settings, client=client
    ))
    assert result["position"] == 10
    assert result["ref"] == "C" and result["alt"] == "G"

    wt_total = expected_total(SEQ, logits)
    mut_seq = SEQ[:9] + "G" + SEQ[10:]
    mut_total = expected_total(mut_seq, logits)
    assert result["wildtype_log_likelihood"] == pytest.approx(wt_total)
    assert result["mutant_log_likelihood"] == pytest.approx(mut_total)
    assert result["delta_log_likelihood"] == pytest.approx(mut_total - wt_total)
    assert result["delta_log_likelihood"] == pytest.approx(-2.0)

    # Interpretation must NOT claim pathogenicity.
    assert "less likely" in result["interpretation"]
    assert "NOT a clinical pathogenicity call" in result["interpretation"]
    assert "clinical" in result["disclaimer"].lower()


def test_variant_score_zero_based_coordinate(settings):
    logits = next_base_logits(SEQ)
    client = FakeEvo2Client({"output_layer": logits})
    r1 = asyncio.run(evo2_variant_score(
        sequence=SEQ, position=10, ref="C", alt="G", coordinate="1-based",
        settings=settings, client=client,
    ))
    r0 = asyncio.run(evo2_variant_score(
        sequence=SEQ, position=9, ref="C", alt="G", coordinate="0-based",
        settings=settings, client=client,
    ))
    assert r0["delta_log_likelihood"] == pytest.approx(r1["delta_log_likelihood"])
    assert r1["delta_log_likelihood"] == pytest.approx(-2.0)


def test_variant_score_makes_two_calls(settings):
    logits = uniform_logits(len(SEQ))
    client = FakeEvo2Client({"output_layer": logits})
    asyncio.run(evo2_variant_score(
        sequence=SEQ, position=10, ref="C", alt="G", settings=settings, client=client
    ))
    assert len(client.calls) == 2  # WT + mutant


def test_variant_score_no_delta_for_uniform(settings):
    logits = uniform_logits(len(SEQ))
    client = FakeEvo2Client({"output_layer": logits})
    result = asyncio.run(evo2_variant_score(
        sequence=SEQ, position=10, ref="C", alt="G", settings=settings, client=client
    ))
    assert result["delta_log_likelihood"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# evo2_batch_score
# ---------------------------------------------------------------------------

def test_batch_score_dedup_and_wt_once(settings):
    logits = uniform_logits(len(SEQ))
    client = FakeEvo2Client({"output_layer": logits})
    variants = [
        {"position": 10, "ref": "C", "alt": "G"},
        {"position": 12, "ref": "T", "alt": "C"},
        {"position": 10, "ref": "C", "alt": "G"},   # duplicate of #0
        {"position": 10, "ref": "C", "alt": "T"},   # same position, different alt
    ]
    result = asyncio.run(evo2_batch_score(
        sequence=SEQ, variants=variants, settings=settings, client=client
    ))
    assert len(result["variants"]) == 4
    # WT once + 3 unique mutants (G at idx9, C at idx11, T at idx9)
    assert len(client.calls) == 1 + 3
    # order preserved
    assert [v["position"] for v in result["variants"]] == [10, 12, 10, 10]
    assert [v["alt"] for v in result["variants"]] == ["G", "C", "G", "T"]
    # duplicates carry identical scores
    assert result["variants"][0]["delta_log_likelihood"] == pytest.approx(
        result["variants"][2]["delta_log_likelihood"]
    )
    # uniform logits -> no delta (all positions equally likely)
    for v in result["variants"]:
        assert v["delta_log_likelihood"] == pytest.approx(0.0, abs=1e-6)


def test_batch_score_delta_magnitude(settings):
    logits = next_base_logits(SEQ)
    client = FakeEvo2Client({"output_layer": logits})
    result = asyncio.run(evo2_batch_score(
        sequence=SEQ,
        variants=[{"position": 10, "ref": "C", "alt": "G"}],
        settings=settings, client=client,
    ))
    assert result["variants"][0]["delta_log_likelihood"] == pytest.approx(-2.0)
    assert result["wildtype_log_likelihood"] == pytest.approx(expected_total(SEQ, logits))


def test_batch_score_validation_fails_fast(settings):
    client = FakeEvo2Client({"output_layer": uniform_logits(len(SEQ))})
    with pytest.raises(SequenceValidationError):
        # position 5 (1-based) -> index 4 = 'A', so ref='T' mismatches.
        asyncio.run(evo2_batch_score(
            sequence=SEQ,
            variants=[{"position": 5, "ref": "T", "alt": "A"}],
            settings=settings, client=client,
        ))
    assert client.calls == []


def test_batch_score_missing_field(settings):
    client = FakeEvo2Client({"output_layer": uniform_logits(len(SEQ))})
    with pytest.raises(ValueError, match="missing field"):
        asyncio.run(evo2_batch_score(
            sequence=SEQ, variants=[{"position": 10, "ref": "C"}],  # alt missing
            settings=settings, client=client,
        ))


def test_batch_score_isolates_per_variant_errors(settings):
    class FlakyClient:
        def __init__(self):
            self.calls = []
            self.logits_layer = "output_layer"

        async def forward_logits(self, *, sequence):
            return await self.forward(sequence=sequence, output_layers=[self.logits_layer])

        async def forward(self, *, sequence, output_layers):
            self.calls.append(sequence)
            if "CGCACG" in sequence:  # unique to the T->C mutant at index 11
                raise Evo2APIError(429, "rate limited")
            import io
            buf = io.BytesIO()
            np.savez(buf, output_layer=uniform_logits(len(SEQ)))
            return buf.getvalue(), {"elapsed_ms": 1}

    client = FlakyClient()
    variants = [
        {"position": 10, "ref": "C", "alt": "G"},
        {"position": 12, "ref": "T", "alt": "C"},  # -> 429
        {"position": 14, "ref": "C", "alt": "T"},
    ]
    result = asyncio.run(evo2_batch_score(
        sequence=SEQ, variants=variants, settings=settings, client=client
    ))
    entries = {v["position"]: v for v in result["variants"]}
    assert "delta_log_likelihood" in entries[10]
    assert entries[12]["error"] and entries[12]["status"] == 429
    assert "delta_log_likelihood" in entries[14]


# ---------------------------------------------------------------------------
# evo2_score_fasta
# ---------------------------------------------------------------------------

FASTA_TEXT = ">geneA\nACGTACGTACGT\n>geneB\nTTGGCCAATTGG\n"


def test_fasta_inline(settings):
    logits = uniform_logits(12)
    client = FakeEvo2Client({"output_layer": logits})
    result = asyncio.run(evo2_score_fasta(
        fasta_text=FASTA_TEXT, settings=settings, client=client
    ))
    assert result["num_records"] == 2
    rA, rB = result["records"]
    assert rA["id"] == "geneA"
    assert rA["total_log_likelihood"] == pytest.approx(11 * -LOG_512)
    assert rB["id"] == "geneB"
    assert rA["sequence_length"] == 12


def test_fasta_path_denied_without_allowed_dirs(settings):
    client = FakeEvo2Client({"output_layer": uniform_logits(4)})
    with pytest.raises(FastaError, match="disabled"):
        asyncio.run(evo2_score_fasta(
            fasta_path="/tmp/somewhere/genes.fa", settings=settings, client=client
        ))


def test_fasta_path_respects_allowed_dirs(settings, tmp_path):
    allowed = tmp_path / "genomes"
    allowed.mkdir()
    fasta = allowed / "genes.fa"
    fasta.write_text(FASTA_TEXT)
    settings = settings.__class__(**{
        **settings.__dict__,
        "allowed_dirs": (allowed.resolve(),),
    })
    client = FakeEvo2Client({"output_layer": uniform_logits(12)})
    result = asyncio.run(evo2_score_fasta(
        fasta_path=str(fasta), settings=settings, client=client
    ))
    assert result["num_records"] == 2


def test_fasta_path_outside_allowed_dirs(settings, tmp_path):
    allowed = tmp_path / "genomes"
    allowed.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    fasta = other / "genes.fa"
    fasta.write_text(FASTA_TEXT)
    settings = settings.__class__(**{
        **settings.__dict__,
        "allowed_dirs": (allowed.resolve(),),
    })
    client = FakeEvo2Client({"output_layer": uniform_logits(12)})
    with pytest.raises(FastaError, match="Refusing to read"):
        asyncio.run(evo2_score_fasta(
            fasta_path=str(fasta), settings=settings, client=client
        ))


def test_fasta_exactly_one_source_required(settings):
    client = FakeEvo2Client({"output_layer": uniform_logits(4)})
    with pytest.raises(ValueError, match="exactly one"):
        asyncio.run(evo2_score_fasta(settings=settings, client=client))


def test_fasta_per_record_error_is_isolated(settings):
    text = ">bad\nACGTXACGT\n>good\nACGTACGTACGT\n"
    client = FakeEvo2Client({"output_layer": uniform_logits(12)})
    result = asyncio.run(evo2_score_fasta(fasta_text=text, settings=settings, client=client))
    assert result["records"][0]["error"]
    assert "total_log_likelihood" in result["records"][1]
