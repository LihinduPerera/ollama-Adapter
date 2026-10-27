from __future__ import annotations

import httpx
import pytest

from adapter.errors import UpstreamError
from adapter.ollama_client import OllamaClient


def make_client(handler, *, max_response_bytes: int = 8_000_000) -> OllamaClient:
    return OllamaClient(
        "http://test",
        timeout=10.0,
        max_response_bytes=max_response_bytes,
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_chat_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "hello"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3,
                "eval_count": 2,
            },
        )

    client = make_client(handler)
    try:
        result = await client.chat({"model": "m", "messages": [], "stream": False})
    finally:
        await client.close()
    assert result["message"]["content"] == "hello"


@pytest.mark.asyncio
async def test_chat_maps_404() -> None:
    client = make_client(
        lambda request: httpx.Response(404, json={"error": "model 'x' not found"})
    )
    try:
        with pytest.raises(UpstreamError) as exc_info:
            await client.chat({"model": "x", "messages": []})
    finally:
        await client.close()
    assert exc_info.value.status_code == 404
    assert exc_info.value.code == "model_not_found"


@pytest.mark.asyncio
async def test_chat_maps_400() -> None:
    client = make_client(
        lambda request: httpx.Response(400, json={"error": "bad request"})
    )
    try:
        with pytest.raises(UpstreamError) as exc_info:
            await client.chat({"model": "x", "messages": []})
    finally:
        await client.close()
    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "ollama_bad_request"


@pytest.mark.asyncio
async def test_chat_rejects_invalid_json() -> None:
    client = make_client(lambda request: httpx.Response(200, content=b"not json"))
    try:
        with pytest.raises(UpstreamError) as exc_info:
            await client.chat({"model": "x", "messages": []})
    finally:
        await client.close()
    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_chat_rejects_missing_message() -> None:
    client = make_client(
        lambda request: httpx.Response(200, json={"done": True})
    )
    try:
        with pytest.raises(UpstreamError) as exc_info:
            await client.chat({"model": "x", "messages": []})
    finally:
        await client.close()
    assert "missing a message" in exc_info.value.message


@pytest.mark.asyncio
async def test_chat_rejects_oversized_response() -> None:
    client = make_client(
        lambda request: httpx.Response(200, content=b"x" * 100),
        max_response_bytes=10,
    )
    try:
        with pytest.raises(UpstreamError) as exc_info:
            await client.chat({"model": "x", "messages": []})
    finally:
        await client.close()
    assert exc_info.value.code == "ollama_response_too_large"


@pytest.mark.asyncio
async def test_chat_maps_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = make_client(handler)
    try:
        with pytest.raises(UpstreamError) as exc_info:
            await client.chat({"model": "x", "messages": []})
    finally:
        await client.close()
    assert exc_info.value.status_code == 503
    assert exc_info.value.code == "ollama_unreachable"


@pytest.mark.asyncio
async def test_chat_maps_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client = make_client(handler)
    try:
        with pytest.raises(UpstreamError) as exc_info:
            await client.chat({"model": "x", "messages": []})
    finally:
        await client.close()
    assert exc_info.value.status_code == 504
    assert exc_info.value.code == "ollama_timeout"


@pytest.mark.asyncio
async def test_health_success() -> None:
    client = make_client(
        lambda request: httpx.Response(200, json={"version": "0.40.11"})
    )
    try:
        health = await client.health()
    finally:
        await client.close()
    assert health == {"reachable": True, "version": "0.40.11"}


@pytest.mark.asyncio
async def test_health_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = make_client(handler)
    try:
        health = await client.health()
    finally:
        await client.close()
    assert health == {"reachable": False}
