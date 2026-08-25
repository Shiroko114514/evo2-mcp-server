"""Server-level tests: tool schemas + a full MCP session over in-memory
streams with a mocked Evo2Client (no real HTTP)."""

from __future__ import annotations

import asyncio
import io

import numpy as np
import pytest

from evo2_mcp.server import _SAFETY_NOTE, _tool_schemas, build_server

from conftest import uniform_logits


def test_tool_schemas_present():
    schemas = _tool_schemas()
    names = [t.name for t in schemas]
    assert names == [
        "evo2_forward",
        "evo2_score",
        "evo2_variant_score",
        "evo2_batch_score",
        "evo2_score_fasta",
    ]
    for t in schemas:
        # Every agent-facing description carries the safety disclaimer.
        assert "clinical" in t.description
        assert "pathogenicity" in t.description
    fwd = next(t for t in schemas if t.name == "evo2_forward")
    assert "output_layers" in fwd.input_schema["properties"]
    assert fwd.input_schema["required"] == ["sequence"]


class FakeEvo2Client:
    def __init__(self, settings):
        self.settings = settings
        self.logits_layer = "output_layer"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def forward_logits(self, *, sequence):
        return await self.forward(sequence=sequence, output_layers=[self.logits_layer])

    async def forward(self, *, sequence, output_layers):
        buf = io.BytesIO()
        np.savez(buf, output_layer=uniform_logits(len(sequence)))
        return buf.getvalue(), {"elapsed_ms": 3}


async def _run_session(settings, monkeypatch):
    from mcp import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    monkeypatch.setattr("evo2_mcp.server.Evo2Client", FakeEvo2Client)
    server = build_server(settings)

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        server_task = asyncio.create_task(
            server.run(
                server_streams[0],
                server_streams[1],
                server.create_initialization_options(),
            )
        )
        try:
            async with ClientSession(client_streams[0], client_streams[1]) as session:
                await session.initialize()
                tools = await session.list_tools()
                return tools
        finally:
            server_task.cancel()
            try:
                await server_task
            except (asyncio.CancelledError, Exception):
                pass


@pytest.mark.asyncio
async def test_session_list_tools(settings, monkeypatch):
    tools = await _run_session(settings, monkeypatch)
    assert {t.name for t in tools.tools} == {
        "evo2_forward",
        "evo2_score",
        "evo2_variant_score",
        "evo2_batch_score",
        "evo2_score_fasta",
    }


@pytest.mark.asyncio
async def test_session_call_evo2_score(settings, monkeypatch):
    from mcp import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    monkeypatch.setattr("evo2_mcp.server.Evo2Client", FakeEvo2Client)
    server = build_server(settings)

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        server_task = asyncio.create_task(
            server.run(
                server_streams[0],
                server_streams[1],
                server.create_initialization_options(),
            )
        )
        try:
            async with ClientSession(client_streams[0], client_streams[1]) as session:
                await session.initialize()
                result = await session.call_tool(
                    "evo2_score",
                    {"sequence": "ACGTACGTACGT"},
                )
                import json
                payload = json.loads(result.content[0].text)
                assert payload["likelihood_computed"] is True
                assert payload["scored_positions"] == 11
                assert payload["total_log_likelihood"] == pytest.approx(11 * -np.log(512.0))
        finally:
            server_task.cancel()
            try:
                await server_task
            except (asyncio.CancelledError, Exception):
                pass


@pytest.mark.asyncio
async def test_session_error_is_structured_json(settings, monkeypatch):
    from mcp import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    monkeypatch.setattr("evo2_mcp.server.Evo2Client", FakeEvo2Client)
    server = build_server(settings)

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        server_task = asyncio.create_task(
            server.run(
                server_streams[0],
                server_streams[1],
                server.create_initialization_options(),
            )
        )
        try:
            async with ClientSession(client_streams[0], client_streams[1]) as session:
                await session.initialize()
                # invalid sequence -> structured error, not a traceback
                result = await session.call_tool("evo2_score", {"sequence": "ACGTX"})
                import json
                payload = json.loads(result.content[0].text)
                assert payload["error"] == "SequenceValidationError"
                assert "X" in payload["message"]
        finally:
            server_task.cancel()
            try:
                await server_task
            except (asyncio.CancelledError, Exception):
                pass
