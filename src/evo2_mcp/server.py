"""MCP server exposing Evo2 tools via stdio.

Built on the official Python MCP SDK (>= 2.0) low-level `mcp.server.Server`
API: tool schemas are declared explicitly so MCP clients (Claude Code,
Cursor, Codex, ...) can read them to decide when to call each tool.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import mcp.types as mcp_types
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server

from evo2_mcp.api_client import Evo2APIError, Evo2Client
from evo2_mcp.config import Evo2ConfigError, Settings
from evo2_mcp.fasta import FastaError
from evo2_mcp.forward_output import ForwardDecodeError
from evo2_mcp.sequence import SequenceValidationError
from evo2_mcp.tools import (
    evo2_batch_score,
    evo2_forward,
    evo2_score,
    evo2_score_fasta,
    evo2_variant_score,
)
from evo2_mcp.version import __version__

logger = logging.getLogger("evo2_mcp.server")

_SAFETY_NOTE = (
    "This is a DNA foundation model inference tool. It does not provide "
    "clinical diagnosis. Model scores should not be interpreted as "
    "pathogenicity labels without additional validation."
)


def _tool_schemas() -> list[mcp_types.Tool]:
    return [
        mcp_types.Tool(
            name="evo2_forward",
            description=(
                "Run a forward pass of Evo2-7B on a DNA sequence (forward "
                "inference) and return summary statistics — or raw tensors — "
                "for the requested model layers (final logits, attention, MLP "
                "or embedding outputs). Use this when you need layer outputs "
                "for analysis, not just a scalar score. Modes: 'summary' "
                "(shape/dtype/min/max/mean/std per layer — default, "
                "context-safe), 'save' (write the .npz under the server's "
                "output dir and return the path), 'raw' (inline small tensors "
                "only; large tensors must be saved to a file instead). "
                f"{_SAFETY_NOTE}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": (
                            "DNA sequence over {A,C,G,T}. Lowercase and "
                            "whitespace are normalised. IUPAC ambiguity codes "
                            "(N, R, Y, ...) are rejected unless the server was "
                            "started with EVO2_MCP_ALLOW_AMBIGUOUS=1 (N only)."
                        ),
                    },
                    "output_layers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Evo2 layer names. The final logits layer is "
                            "'output_layer' on self-hosted NIM 2.x containers, "
                            "but the hosted health.api.nvidia.com endpoint "
                            "uses model attribute names — use 'unembed' for "
                            "final logits (also 'embedding_layer', 'norm', "
                            "'blocks.N.mlp'). Scoring tools auto-detect this. "
                            "See https://docs.nvidia.com/nim/bionemo/evo2/latest/endpoints.html"
                        ),
                        "default": ["output_layer"],
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["summary", "raw", "save"],
                        "default": "summary",
                        "description": (
                            "summary: per-layer shape/dtype/min/max/mean/std "
                            "only. raw: inline the tensors as nested lists "
                            "(small only, capped). save: always write an .npz "
                            "file in the output dir and return its path."
                        ),
                    },
                    "save_path": {
                        "type": ["string", "null"],
                        "description": (
                            "Optional filename for the saved NPZ (implies "
                            "mode='save'). Must resolve inside the server's "
                            "output directory (EVO2_MCP_OUTPUT_DIR, default "
                            "./output). If omitted, a timestamped file is used."
                        ),
                    },
                },
                "required": ["sequence"],
                "additionalProperties": False,
            },
        ),
        mcp_types.Tool(
            name="evo2_score",
            description=(
                "Compute the model-based log-likelihood of a DNA sequence "
                "under Evo2-7B (nucleotide-level log-likelihood via the "
                "byte-level tokenizer). Returns total_log_likelihood, "
                "mean_log_likelihood, scored_positions and optionally "
                "per_position_log_likelihood (position 0 is unscored by the "
                "causal shift; value k corresponds to 0-based position k+1). "
                "Use for sequence-level probability estimates. "
                f"{_SAFETY_NOTE}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "DNA sequence over {A,C,G,T} (>= 2 bp).",
                    },
                    "include_per_position": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Return the per-position log-likelihood list "
                            "(default False to keep the response small; long "
                            "lists are truncated to head+tail)."
                        ),
                    },
                },
                "required": ["sequence"],
                "additionalProperties": False,
            },
        ),
        mcp_types.Tool(
            name="evo2_variant_score",
            description=(
                "Compare a single-nucleotide variant: the Evo2-7B "
                "log-likelihood of the wildtype sequence vs the mutant "
                "sequence. Returns delta_log_likelihood "
                "(mutant − wildtype); a negative value means the mutant "
                "sequence is LESS likely under the model. Positions are "
                "1-based by default (VCF-style). Variants at position 1 are "
                "rejected because a causal LM cannot score the first base. "
                f"{_SAFETY_NOTE}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Wildtype DNA sequence context (>= 2 bp).",
                    },
                    "position": {
                        "type": "integer",
                        "description": "Variant position, 1-based by default.",
                    },
                    "ref": {
                        "type": "string",
                        "description": "Reference allele (single A/C/G/T).",
                    },
                    "alt": {
                        "type": "string",
                        "description": "Alternate allele (single A/C/G/T).",
                    },
                    "coordinate": {
                        "type": "string",
                        "enum": ["1-based", "0-based"],
                        "default": "1-based",
                        "description": "Coordinate system of `position`.",
                    },
                    "include_per_position": {
                        "type": "boolean",
                        "default": False,
                    },
                },
                "required": ["sequence", "position", "ref", "alt"],
                "additionalProperties": False,
            },
        ),
        mcp_types.Tool(
            name="evo2_batch_score",
            description=(
                "Score many single-nucleotide variants against one wildtype "
                "sequence. The WT forward pass is computed exactly once and "
                "reused; identical (position, alt) mutants are forwarded once; "
                "mutant requests run with bounded concurrency "
                "(EVO2_MCP_MAX_CONCURRENCY, default 2) to respect NVIDIA rate "
                "limits, and per-variant API errors are reported per-variant. "
                "Use for saturation-mutagenesis-style analyses. "
                f"{_SAFETY_NOTE}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sequence": {"type": "string", "description": "Wildtype DNA sequence (>= 2 bp)."},
                    "variants": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "position": {"type": "integer"},
                                "ref": {"type": "string"},
                                "alt": {"type": "string"},
                            },
                            "required": ["position", "ref", "alt"],
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                        "description": "List of {position, ref, alt} dicts (1-based positions).",
                    },
                    "coordinate": {
                        "type": "string",
                        "enum": ["1-based", "0-based"],
                        "default": "1-based",
                    },
                },
                "required": ["sequence", "variants"],
                "additionalProperties": False,
            },
        ),
        mcp_types.Tool(
            name="evo2_score_fasta",
            description=(
                "Score every record in a FASTA source under Evo2-7B. Provide "
                "EITHER `fasta_text` (inline FASTA) OR `fasta_path` (local "
                "file — only allowed when its directory is listed in "
                "EVO2_MCP_ALLOWED_DIRS; the server refuses arbitrary paths). "
                "Returns per-record total/mean log-likelihood; per-record "
                "errors are reported inline. "
                f"{_SAFETY_NOTE}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "fasta_path": {
                        "type": ["string", "null"],
                        "description": "Local FASTA path (sandboxed by EVO2_MCP_ALLOWED_DIRS).",
                    },
                    "fasta_text": {
                        "type": ["string", "null"],
                        "description": "Inline FASTA text, e.g. '>seq1\\nACGT...'.",
                    },
                },
                "additionalProperties": False,
            },
        ),
    ]


def build_server(settings: Settings) -> Server:
    async def _list_tools(
        ctx: ServerRequestContext,
        params: mcp_types.PaginatedRequestParams | None,
    ) -> mcp_types.ListToolsResult:
        return mcp_types.ListToolsResult(tools=_tool_schemas())

    async def _call_tool(
        ctx: ServerRequestContext,
        params: mcp_types.CallToolRequestParams,
    ) -> mcp_types.CallToolResult:
        name = params.name
        arguments = dict(params.arguments or {})
        try:
            async with Evo2Client(settings) as client:
                if name == "evo2_forward":
                    result = await evo2_forward(**arguments, settings=settings, client=client)
                elif name == "evo2_score":
                    result = await evo2_score(**arguments, settings=settings, client=client)
                elif name == "evo2_variant_score":
                    result = await evo2_variant_score(**arguments, settings=settings, client=client)
                elif name == "evo2_batch_score":
                    result = await evo2_batch_score(**arguments, settings=settings, client=client)
                elif name == "evo2_score_fasta":
                    result = await evo2_score_fasta(**arguments, settings=settings, client=client)
                else:
                    raise ValueError(f"Unknown tool: {name}")
        except (
            Evo2ConfigError,
            Evo2APIError,
            SequenceValidationError,
            ForwardDecodeError,
            FastaError,
            ValueError,
        ) as e:
            payload = {"error": type(e).__name__, "message": str(e)}
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text=json.dumps(payload, indent=2))],
                is_error=True,
            )

        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, default=_json_default),
            )]
        )

    return Server(
        name="evo2-mcp",
        version=__version__,
        on_list_tools=_list_tools,
        on_call_tool=_call_tool,
    )


def _json_default(o: Any) -> Any:
    # Rarely used; helpers return json-safe values, but be safe with numpy scalars.
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    return str(o)


async def _run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    settings = Settings.from_env()
    server = build_server(settings)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    import asyncio
    asyncio.run(_run())


if __name__ == "__main__":
    main()
