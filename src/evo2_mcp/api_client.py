"""HTTP client for the NVIDIA-hosted Evo2-7B Forward endpoint.

Verified API contract (2026-08-24, official sources):
* Request:  ``POST {base_url}/forward``
  Body: ``{"sequence": <UPPERCASE DNA>, "output_layers": [<layer_name>, ...]}``
  (hosted OpenAPI schema: ForwardInputs = sequence:str(minLength 1),
  output_layers: 1..100 strings; auth: ``Authorization: Bearer $NVIDIA_API_KEY``)
* Response (hosted OpenAPI schema ForwardOutputs):
  JSON ``{"data": <base64 NPZ>, "elapsed_ms": int}``; the NIM can also answer
  with ``Content-Type: application/zip`` (raw NPZ bytes) for large payloads.
  Each requested layer is an array inside the NPZ; ``output_layer`` is the
  final logits of shape ``[seq_len, batch_size, 512]`` (512 = padded vocab).

Error handling:
* 401/403/404/413/422 → terminal, clear agent-friendly message.
* 408/429/5xx → exponential backoff with jitter, capped at `max_retries`;
  ``Retry-After`` (if present) is honoured and capped at 60 s.
* timeouts → `Evo2TimeoutError` with a clear message, retried like 5xx.

We never log the API key or the raw DNA sequence.
"""

from __future__ import annotations

import asyncio
import io
import logging
import random
from typing import Any, Iterable

import httpx
import numpy as np

from evo2_mcp.config import Settings

logger = logging.getLogger("evo2_mcp.api_client")

# Layer name used to request the final logits. The documented NIM name is
# 'output_layer'; the hosted health.api.nvidia.com endpoint (vortex-backed)
# rejects it with `StripedHyena has no attribute 'output_layer'` and expects
# the model attribute name 'unembed' instead (verified live 2026-08-24).
LOGITS_LAYER_FALLBACKS = ("output_layer", "unembed")


class Evo2APIError(RuntimeError):
    def __init__(self, status: int | None, message: str, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


class Evo2TimeoutError(Evo2APIError):
    def __init__(self, message: str):
        super().__init__(None, message)


def _classify(status: int, body_snippet: str) -> Evo2APIError:
    """Turn an HTTP status into a clear, agent-friendly Evo2APIError."""
    if status == 400:
        msg = f"NVIDIA Evo2 API returned 400 Bad Request: {body_snippet}"
    elif status == 401:
        msg = (
            "NVIDIA Evo2 API returned 401 Unauthorized. "
            "Check that NVIDIA_API_KEY is set and valid "
            "(get one from https://build.nvidia.com/)."
        )
    elif status == 403:
        msg = (
            "NVIDIA Evo2 API returned 403 Forbidden. Your key may not have "
            "access to arc/evo2-7b (the hosted endpoint was marked deprecated "
            "on build.nvidia.com in 2026)."
        )
    elif status == 404:
        msg = (
            "NVIDIA Evo2 API returned 404 Not Found. Verify EVO2_MCP_BASE_URL "
            "and that the /forward path exists on the target NIM."
        )
    elif status == 408:
        msg = "NVIDIA Evo2 API returned 408 Request Timeout (server side)."
    elif status == 413:
        msg = (
            "NVIDIA Evo2 API returned 413 Payload Too Large. The DNA sequence "
            "or the requested output_layers are too big for a single request."
        )
    elif status == 422:
        msg = f"NVIDIA Evo2 API returned 422 Validation Error: {body_snippet}"
        if "has no attribute" in body_snippet:
            msg += (
                " Hint: the hosted health.api.nvidia.com endpoint expects Evo2 "
                "model attribute names, e.g. output_layers=['unembed'] for the "
                "final logits (also: 'embedding_layer', 'norm', 'blocks.N.mlp'). "
                "The documented names 'output_layer'/'decoder.layers.N.*' only "
                "work on self-hosted NIM 2.x containers."
            )
    elif status == 429:
        msg = "NVIDIA Evo2 API returned 429 Too Many Requests (rate limited)."
    elif 500 <= status <= 599:
        msg = f"NVIDIA Evo2 API server error {status}: {body_snippet}"
    else:
        msg = f"NVIDIA Evo2 API returned HTTP {status}: {body_snippet}"
    return Evo2APIError(status, msg, body_snippet)


def _should_retry(status: int) -> bool:
    # 408 (server-side timeout) and 429 (rate limit) are transient; 5xx are
    # NVIDIA-side failures. All other 4xx are terminal.
    return status in (408, 429) or 500 <= status <= 599


def _backoff_seconds(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(retry_after, 60.0)
    # Exponential backoff with jitter: 1s, 2s, 4s, 8s, 16s, capped at 30s.
    base = min(2 ** attempt, 30)
    return base + random.uniform(0, 1)


def _logits_alt_name(layer: str) -> str | None:
    """Return the alternative logits layer name, or None if none applies."""
    if layer not in LOGITS_LAYER_FALLBACKS:
        return None
    return "unembed" if layer == "output_layer" else "output_layer"


class Evo2Client:
    """Async client around httpx.AsyncClient for the Evo2 forward endpoint."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self._client = client
        self._owns_client = client is None
        # Resolved logits layer name (auto-detected on first use). `None` =
        # not resolved yet.
        self._logits_layer: str | None = None

    async def __aenter__(self) -> "Evo2Client":
        if self._client is None:
            # Short connect/pool timeouts; the read timeout is the user knob
            # (long sequences can take minutes on the server).
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=15.0,
                    read=self.settings.timeout,
                    write=60.0,
                    pool=15.0,
                )
            )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Evo2Client used outside of `async with`")
        return self._client

    async def forward_logits(self, *, sequence: str) -> tuple[bytes, dict[str, Any]]:
        """Forward requesting only the final-logits layer.

        Layer name resolution (verified live against the hosted API):
        * ``EVO2_MCP_LOGITS_LAYER=auto`` (default): try ``output_layer`` (the
          documented NIM name); if the server rejects it with
          ``has no attribute`` (the hosted vortex-backed endpoint), retry once
          with ``unembed`` and cache the working name for this client.
        * explicit name: used as-is.
        """
        if self._logits_layer is None:
            raw = self.settings.logits_layer
            if raw == "auto":
                self._logits_layer = LOGITS_LAYER_FALLBACKS[0]
            else:
                self._logits_layer = raw

        try:
            npz, meta = await self.forward(sequence=sequence, output_layers=[self._logits_layer])
        except Evo2APIError as e:
            alt = _logits_alt_name(self._logits_layer)
            if alt is not None and e.status == 422 and "has no attribute" in (e.body or ""):
                logger.info(
                    "Logits layer %r rejected by server ('has no attribute'); "
                    "switching to %r for this client.",
                    self._logits_layer, alt,
                )
                self._logits_layer = alt
                npz, meta = await self.forward(sequence=sequence, output_layers=[alt])
            else:
                raise
        return npz, meta

    @property
    def logits_layer(self) -> str | None:
        """Resolved logits layer name (None until first forward_logits call)."""
        return self._logits_layer

    async def forward(
        self,
        *,
        sequence: str,
        output_layers: Iterable[str],
    ) -> tuple[bytes, dict[str, Any]]:
        """Send a forward request. Returns (npz_bytes, response_metadata).

        `response_metadata` includes `elapsed_ms` when the API returned JSON.
        `npz_bytes` is the raw NPZ archive; the caller decodes it with
        `forward_output.decode_npz`.
        """
        api_key = self.settings.require_api_key()
        url = f"{self.settings.base_url}/forward"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "sequence": sequence,
            "output_layers": list(output_layers),
        }

        # Log a redacted summary — never the sequence, never the key.
        logger.info(
            "POST %s (seq_len=%d, output_layers=%s)",
            url,
            len(sequence),
            payload["output_layers"],
        )

        last_error: Evo2APIError | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                response = await self.client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as e:
                last_error = Evo2TimeoutError(
                    f"NVIDIA Evo2 API request timed out after {self.settings.timeout}s "
                    f"(attempt {attempt + 1}/{self.settings.max_retries + 1}): {e}"
                )
                if attempt >= self.settings.max_retries:
                    raise last_error
                await asyncio.sleep(_backoff_seconds(attempt, None))
                continue
            except httpx.HTTPError as e:
                # Network-level error (connection reset, DNS, ...). Retry.
                last_error = Evo2APIError(None, f"HTTP transport error: {e}")
                if attempt >= self.settings.max_retries:
                    raise last_error
                await asyncio.sleep(_backoff_seconds(attempt, None))
                continue

            if response.status_code == 200:
                return _decode_response(response)

            # Non-200: capture a short snippet for diagnostics (never the
            # whole possibly-large body), then decide whether to retry.
            body_snippet = response.text[:500] if response.text else ""
            error = _classify(response.status_code, body_snippet)
            if _should_retry(response.status_code) and attempt < self.settings.max_retries:
                retry_after_raw = response.headers.get("retry-after")
                try:
                    retry_after = float(retry_after_raw) if retry_after_raw else None
                except ValueError:
                    retry_after = None
                delay = _backoff_seconds(attempt, retry_after)
                logger.warning(
                    "Evo2 API HTTP %d, retrying in %.1fs (attempt %d/%d)",
                    response.status_code, delay, attempt + 1, self.settings.max_retries + 1,
                )
                await asyncio.sleep(delay)
                last_error = error
                continue

            raise error

        # Should be unreachable — either we returned 200 or raised above.
        raise last_error or Evo2APIError(None, "Evo2 forward failed for an unknown reason")


def _decode_response(response: httpx.Response) -> tuple[bytes, dict[str, Any]]:
    """Return (npz_bytes, metadata) from a 200 response.

    Handles, in order:
    1. ``application/zip`` → raw NPZ bytes.
    2. JSON with a base64 ``data`` field → the documented hosted response
       (ForwardOutputs).
    3. JSON with a legacy per-layer ``{shape, dtype, data}`` structure →
       re-encoded into NPZ bytes so callers see one uniform format.
    """
    content_type = response.headers.get("Content-Type", "")
    if "application/zip" in content_type:
        return response.content, {"content_type": content_type}

    try:
        body = response.json()
    except Exception as e:
        raise Evo2APIError(
            200,
            f"Evo2 forward returned Content-Type={content_type!r} but body was not valid JSON: {e}",
            response.text[:500],
        )
    if not isinstance(body, dict):
        raise Evo2APIError(200, "Evo2 forward response JSON is not an object.", str(body)[:500])

    if "data" in body:
        import base64
        try:
            npz_bytes = base64.b64decode(body["data"], validate=False)
        except Exception as e:
            raise Evo2APIError(200, f"Failed to base64-decode 'data' field: {e}", None)
        meta = {k: v for k, v in body.items() if k != "data"}
        meta["content_type"] = content_type
        meta["encoded_bytes"] = len(body["data"])
        return npz_bytes, meta

    if isinstance(body.get("output"), dict):
        # Legacy/defensive fallback: {"output": {layer: {shape, dtype, data}}}.
        arrays: dict[str, np.ndarray] = {}
        for name, spec in body["output"].items():
            if not isinstance(spec, dict) or "data" not in spec:
                raise Evo2APIError(
                    200,
                    f"Evo2 forward response layer {name!r} has an unexpected structure.",
                    str(spec)[:300],
                )
            flat = np.asarray(spec["data"])
            shape = tuple(spec.get("shape", flat.shape))
            if flat.size == 0:
                arrays[name] = flat.reshape(shape)
            else:
                arrays[name] = np.asarray(flat).reshape(shape)
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        meta = {"content_type": content_type, "decoded_as": "legacy-json-tensor"}
        return buf.getvalue(), meta

    raise Evo2APIError(
        200,
        f"Evo2 forward response JSON is missing 'data'. Keys: {list(body)}. "
        "Expected the documented {'data': base64-NPZ, 'elapsed_ms': int} format.",
        None,
    )
