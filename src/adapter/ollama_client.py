from __future__ import annotations

import json
from typing import Any

import httpx

from .errors import UpstreamError


class OllamaClient:
    """Minimal async client for the Ollama native chat API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 300.0,
        max_response_bytes: int = 8_000_000,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_response_bytes = max_response_bytes
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
            headers={"Content-Type": "application/json"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                raw = await self._read_limited(response)
                if response.status_code >= 400:
                    raise self._error_from_response(response.status_code, raw)
                try:
                    data = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise UpstreamError(
                        "Ollama returned invalid JSON", status_code=502
                    ) from exc
                if not isinstance(data, dict):
                    raise UpstreamError(
                        "Ollama returned an unexpected response", status_code=502
                    )
                if data.get("error"):
                    raise UpstreamError(
                        f"Ollama error: {str(data['error'])[:300]}", status_code=502
                    )
                message = data.get("message")
                if not isinstance(message, dict):
                    raise UpstreamError(
                        "Ollama response is missing a message", status_code=502
                    )
                return data
        except httpx.ConnectError as exc:
            raise UpstreamError(
                f"Ollama is not reachable at {self.base_url}",
                status_code=503,
                code="ollama_unreachable",
            ) from exc
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                "Ollama request timed out",
                status_code=504,
                code="ollama_timeout",
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(
                "Ollama request failed",
                status_code=502,
                code="ollama_transport_error",
            ) from exc

    async def _read_limited(self, response: httpx.Response) -> bytes:
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > self.max_response_bytes:
                raise UpstreamError(
                    "Ollama response exceeded the configured size limit",
                    status_code=502,
                    code="ollama_response_too_large",
                )
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _error_from_response(status_code: int, raw: bytes) -> UpstreamError:
        message = "Ollama request failed"
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict) and data.get("error"):
            message = str(data["error"])[:300]

        if status_code == 404:
            return UpstreamError(message, status_code=404, code="model_not_found")
        if status_code == 400:
            return UpstreamError(message, status_code=400, code="ollama_bad_request")
        return UpstreamError(message, status_code=502, code="ollama_error")

    async def health(self, *, timeout: float = 2.0) -> dict[str, Any]:
        try:
            response = await self._client.get("/api/version", timeout=timeout)
        except httpx.HTTPError:
            return {"reachable": False}
        if response.status_code != 200:
            return {"reachable": False, "status_code": response.status_code}
        try:
            data = response.json()
        except ValueError:
            return {"reachable": True, "version": None}
        return {
            "reachable": True,
            "version": data.get("version") if isinstance(data, dict) else None,
        }
