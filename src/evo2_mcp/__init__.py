"""Evo2-7B MCP Server.

An MCP server exposing NVIDIA's hosted Evo2-7B Forward API as bioinformatics
tools: DNA forward inference, per-nucleotide likelihood scoring, and single /
batch variant effect prediction.

Verified API contract (2026-08-24):
* Endpoint: POST https://health.api.nvidia.com/v1/biology/arc/evo2-7b/forward
  (hosted endpoint deprecated on build.nvidia.com; NIM container path
  /biology/arc/evo2/forward is supported via EVO2_MCP_BASE_URL).
* Response: {"data": base64-encoded NPZ, "elapsed_ms": int} (or
  application/zip for large payloads). `output_layer` = final logits of shape
  [seq_len, batch_size, 512]; byte-level tokenizer with A=65, C=67, T=84,
  G=71 (ASCII).
* Likelihood semantics follow ArcInstitute/evo2 `logits_to_logprobs`:
  causal shift (logits[i] predicts token i+1), position 0 unscored.

This package is a client for the API; it does not download or run the Evo2
weights locally.
"""

from evo2_mcp.version import __version__

__all__ = ["__version__"]
