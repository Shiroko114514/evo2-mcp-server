"""Tests for the HTTP client: request construction, error mapping, retries,
timeouts and response decoding. All HTTP is mocked with httpx.MockTransport."""

from __future__ import annotations

import asyncio
import base64
import io
import json

import httpx
import numpy as np
import pytest

from evo2_mcp.api_client import Evo2APIError, Evo2Client, Evo2TimeoutError
from evo2_mcp.config import Evo2ConfigError

from conftest import make_npz_bytes, npz_json_response, npz_zip_response, uniform_logits


def make_client(settings, handler) -> Evo2Client:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(5.0))
    return Evo2Client(settings, client=client)


@pytest.mark.asyncio
async def test_api_key_missing(no_key_settings, fast_settings):
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not be called without a key")

    client = make_client(no_key_settings, handler)
    with pytest.raises(Evo2ConfigError, match="NVIDIA_API_KEY"):
        await client.forward(sequence="ACGT", output_layers=["output_layer"])


@pytest.mark.asyncio
async def test_forward_request(settings, fast_settings):
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["auth"] = request.headers.get("authorization")
        captured["payload"] = request.read()
        return npz_json_response({"output_layer": uniform_logits(4)})

    client = make_client(settings, handler)
    resp_bytes, meta = await client.forward(sequence="ACGT", output_layers=["output_layer"])
    assert captured["method"] == "POST"
    assert captured["url"] == "https://health.api.nvidia.com/v1/biology/arc/evo2-7b/forward"
    assert captured["auth"] == "Bearer test-key-123"
    payload = json.loads(captured["payload"])
    assert payload["sequence"] == "ACGT"
    assert payload["output_layers"] == ["output_layer"]
    assert meta["elapsed_ms"] == 7
    assert resp_bytes.startswith(b"PK")  # NPZ is a zip archive


@pytest.mark.asyncio
async def test_forward_multiple_layers(settings, fast_settings):
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        assert payload["output_layers"] == ["output_layer", "decoder.layers.3.self_attention"]
        return npz_json_response({"output_layer": uniform_logits(2)})

    client = make_client(settings, handler)
    await client.forward(
        sequence="ACGT", output_layers=["output_layer", "decoder.layers.3.self_attention"]
    )


@pytest.mark.asyncio
async def test_http_401(settings, fast_settings):
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"detail": "Invalid API key"})

    client = make_client(settings, handler)
    with pytest.raises(Evo2APIError) as ei:
        await client.forward(sequence="ACGT", output_layers=["output_layer"])
    assert ei.value.status == 401
    assert "401" in str(ei.value)
    assert calls["n"] == 1  # 401 is terminal, no retry


@pytest.mark.asyncio
async def test_http_403(settings, fast_settings):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    client = make_client(settings, handler)
    with pytest.raises(Evo2APIError) as ei:
        await client.forward(sequence="ACGT", output_layers=["output_layer"])
    assert ei.value.status == 403
    assert "403" in str(ei.value)


@pytest.mark.asyncio
async def test_http_404(settings, fast_settings):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    client = make_client(settings, handler)
    with pytest.raises(Evo2APIError) as ei:
        await client.forward(sequence="ACGT", output_layers=["output_layer"])
    assert ei.value.status == 404


@pytest.mark.asyncio
async def test_http_422(settings, fast_settings):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "sequence too long"})

    client = make_client(settings, handler)
    with pytest.raises(Evo2APIError) as ei:
        await client.forward(sequence="ACGT", output_layers=["output_layer"])
    assert ei.value.status == 422
    assert "422" in str(ei.value)


@pytest.mark.asyncio
async def test_http_429_retries_then_succeeds(settings, fast_settings):
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "1"})
        return npz_json_response({"output_layer": uniform_logits(2)})

    client = make_client(settings, handler)
    resp, _ = await client.forward(sequence="ACGT", output_layers=["output_layer"])
    assert resp
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_http_429_exhausts_retries(settings, fast_settings):
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, text="rate limited")

    client = make_client(settings, handler)
    with pytest.raises(Evo2APIError) as ei:
        await client.forward(sequence="ACGT", output_layers=["output_layer"])
    assert ei.value.status == 429
    assert "429" in str(ei.value)
    assert calls["n"] == settings.max_retries + 1  # bounded, not infinite


@pytest.mark.asyncio
async def test_http_500_retries_then_succeeds(settings, fast_settings):
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="nvidia hiccup")
        return npz_json_response({"output_layer": uniform_logits(2)})

    client = make_client(settings, handler)
    resp, _ = await client.forward(sequence="ACGT", output_layers=["output_layer"])
    assert resp
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_http_500_exhausts_retries(settings, fast_settings):
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="server error")

    client = make_client(settings, handler)
    with pytest.raises(Evo2APIError) as ei:
        await client.forward(sequence="ACGT", output_layers=["output_layer"])
    assert ei.value.status == 500
    assert calls["n"] == settings.max_retries + 1


@pytest.mark.asyncio
async def test_http_408_retried(settings, fast_settings):
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(408, text="server timeout")
        return npz_json_response({"output_layer": uniform_logits(2)})

    client = make_client(settings, handler)
    resp, _ = await client.forward(sequence="ACGT", output_layers=["output_layer"])
    assert resp
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_timeout_raises_clear_error(settings, fast_settings):
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("read timed out")

    client = make_client(settings, handler)
    with pytest.raises(Evo2TimeoutError) as ei:
        await client.forward(sequence="ACGT", output_layers=["output_layer"])
    assert "timed out" in str(ei.value).lower()
    assert calls["n"] == settings.max_retries + 1


@pytest.mark.asyncio
async def test_transport_error_retried(settings, fast_settings):
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("connection reset")
        return npz_json_response({"output_layer": uniform_logits(2)})

    client = make_client(settings, handler)
    resp, _ = await client.forward(sequence="ACGT", output_layers=["output_layer"])
    assert resp


@pytest.mark.asyncio
async def test_decode_zip_response(settings, fast_settings):
    async def handler(request: httpx.Request) -> httpx.Response:
        return npz_zip_response({"output_layer": uniform_logits(3)})

    client = make_client(settings, handler)
    npz_bytes, meta = await client.forward(sequence="ACGT", output_layers=["output_layer"])
    assert meta["content_type"] == "application/zip"
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as npz:
        assert npz["output_layer"].shape == (3, 512)


@pytest.mark.asyncio
async def test_decode_legacy_json_tensor(settings, fast_settings):
    """Defensive fallback: {'output': {layer: {shape, dtype, data}}} JSON."""

    async def handler(request: httpx.Request) -> httpx.Response:
        arr = np.zeros((4, 1, 512), dtype=np.float32)
        body = {
            "output": {
                "output_layer": {
                    "shape": [4, 1, 512],
                    "dtype": "float32",
                    "data": arr.ravel().tolist(),
                }
            }
        }
        return httpx.Response(200, json=body, headers={"Content-Type": "application/json"})

    client = make_client(settings, handler)
    npz_bytes, meta = await client.forward(sequence="ACGT", output_layers=["output_layer"])
    assert meta["decoded_as"] == "legacy-json-tensor"
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as npz:
        assert npz["output_layer"].shape == (4, 1, 512)


@pytest.mark.asyncio
async def test_decode_malformed_json(settings, fast_settings):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"not json at all", headers={"Content-Type": "application/json"}
        )

    client = make_client(settings, handler)
    with pytest.raises(Evo2APIError):
        await client.forward(sequence="ACGT", output_layers=["output_layer"])


@pytest.mark.asyncio
async def test_decode_json_missing_data(settings, fast_settings):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"elapsed_ms": 3}, headers={"Content-Type": "application/json"})

    client = make_client(settings, handler)
    with pytest.raises(Evo2APIError, match="missing 'data'"):
        await client.forward(sequence="ACGT", output_layers=["output_layer"])


# ---------------------------------------------------------------------------
# forward_logits: hosted-API layer-name auto-detection
# ---------------------------------------------------------------------------

def _unembed_npz_response(seq_len: int = 4) -> httpx.Response:
    import io
    buf = io.BytesIO()
    np.savez(buf, **{"unembed.output": np.zeros((1, seq_len, 512), dtype=np.float64)})
    payload = {"data": base64.b64encode(buf.getvalue()).decode("ascii"), "elapsed_ms": 5}
    return httpx.Response(200, json=payload, headers={"Content-Type": "application/json"})


@pytest.mark.asyncio
async def test_forward_logits_auto_fallback_hosted(settings, fast_settings):
    """Hosted API: 'output_layer' 422s with 'has no attribute' -> auto-switch
    to 'unembed', cache it, and do not repeat the failed probe."""
    calls: list[list[str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        calls.append(payload["output_layers"])
        if payload["output_layers"] == ["output_layer"]:
            return httpx.Response(
                422, json={"error": "StripedHyena has no attribute `output_layer`"}
            )
        return _unembed_npz_response()

    client = make_client(settings, handler)
    npz_bytes, meta = await client.forward_logits(sequence="ACGT")
    assert calls == [["output_layer"], ["unembed"]]
    assert client.logits_layer == "unembed"
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as npz:
        assert "unembed.output" in npz.files
        assert npz["unembed.output"].shape == (1, 4, 512)

    # cached: the second call goes straight to 'unembed'
    await client.forward_logits(sequence="ACGT")
    assert calls[-1] == ["unembed"]


@pytest.mark.asyncio
async def test_forward_logits_nim_documented_name(settings, fast_settings):
    """Self-hosted NIM: 'output_layer' works first try; no fallback probe."""

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        assert payload["output_layers"] == ["output_layer"]
        return npz_json_response({"output_layer": uniform_logits(4)})

    client = make_client(settings, handler)
    npz_bytes, _ = await client.forward_logits(sequence="ACGT")
    assert client.logits_layer == "output_layer"
    assert npz_bytes


@pytest.mark.asyncio
async def test_forward_logits_explicit_name(settings, fast_settings):
    """EVO2_MCP_LOGITS_LAYER=unembed: no probing, straight to the name."""
    explicit = settings.__class__(**{**settings.__dict__, "logits_layer": "unembed"})
    calls: list[list[str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        calls.append(payload["output_layers"])
        return _unembed_npz_response()

    client = make_client(explicit, handler)
    await client.forward_logits(sequence="ACGT")
    assert calls == [["unembed"]]


@pytest.mark.asyncio
async def test_forward_logits_non_attribute_422_not_swallowed(settings, fast_settings):
    """A 422 that is NOT 'has no attribute' must propagate unchanged."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "sequence too long"})

    client = make_client(settings, handler)
    with pytest.raises(Evo2APIError) as ei:
        await client.forward_logits(sequence="ACGT")
    assert ei.value.status == 422
    assert "has no attribute" not in str(ei.value)
