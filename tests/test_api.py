from __future__ import annotations

import json

import httpx
import pytest

from adapter.config import Settings
from adapter.errors import UpstreamError
from adapter.main import create_app
from adapter.protocol import (
    SAFE_CYCLE,
    SAFE_INVALID,
    SAFE_NO_TOOLS,
    SAFE_REPEAT,
    SAFE_SECURITY,
    SAFE_UNKNOWN,
)
from conftest import OPENCODE_TOOLS, FakeOllamaClient, ollama_response


def chat_body(
    *,
    model: str = "qwen2.5-coder:7b",
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
    stream: bool = False,
    **extra,
) -> dict:
    body = {
        "model": model,
        "messages": messages or [{"role": "user", "content": "hello"}],
        "stream": stream,
    }
    if tools is not None:
        body["tools"] = tools
    body.update(extra)
    return body


@pytest.fixture
async def api(settings):
    fake = FakeOllamaClient()
    app = create_app(settings, ollama_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        yield client, fake


@pytest.mark.asyncio
async def test_non_stream_text_response(api) -> None:
    client, fake = api
    fake.responses.append(ollama_response("Hello there."))
    response = await client.post(
        "/v1/chat/completions",
        json=chat_body(tools=None),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Hello there.",
    }
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert fake.requests[0]["stream"] is False
    assert "tools" not in fake.requests[0]


@pytest.mark.asyncio
async def test_bare_json_tool_call_is_translated(api) -> None:
    client, fake = api
    fake.responses.append(
        ollama_response('{"name":"get_directory","arguments":{"path":"."}}')
    )
    response = await client.post(
        "/v1/chat/completions",
        json=chat_body(tools=OPENCODE_TOOLS),
    )
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    calls = choice["message"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "read"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "."}
    assert calls[0]["id"].startswith("call_")

    upstream = fake.requests[0]
    assert upstream["stream"] is False
    assert upstream["options"]["num_ctx"] == 16384
    assert upstream["tools"][0]["function"]["name"] == "read"


@pytest.mark.asyncio
async def test_structured_tool_call_is_translated(api) -> None:
    client, fake = api
    fake.responses.append(
        ollama_response(
            "",
            tool_calls=[
                {"function": {"name": "read", "arguments": {"path": "a.txt"}}}
            ],
        )
    )
    response = await client.post(
        "/v1/chat/completions", json=chat_body(tools=OPENCODE_TOOLS)
    )
    assert response.status_code == 200
    call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "read"
    assert json.loads(call["function"]["arguments"]) == {"path": "a.txt"}


@pytest.mark.asyncio
async def test_multiple_tool_calls(api) -> None:
    client, fake = api
    fake.responses.append(
        ollama_response(
            '[{"name":"read","arguments":{"path":"a.txt"}},'
            '{"name":"read","arguments":{"path":"b.txt"}}]'
        )
    )
    response = await client.post(
        "/v1/chat/completions", json=chat_body(tools=OPENCODE_TOOLS)
    )
    assert response.status_code == 200
    calls = response.json()["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 2
    assert [json.loads(call["function"]["arguments"])["path"] for call in calls] == [
        "a.txt",
        "b.txt",
    ]


@pytest.mark.asyncio
async def test_streaming_tool_call_shape(api) -> None:
    client, fake = api
    fake.responses.append(
        ollama_response('{"name":"read","arguments":{"path":"a.txt"}}')
    )
    response = await client.post(
        "/v1/chat/completions",
        json=chat_body(
            tools=OPENCODE_TOOLS,
            stream=True,
            stream_options={"include_usage": True},
        ),
        headers={"x-request-id": "req-123"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-request-id"] == "req-123"

    data_lines = [
        line[6:]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert data_lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in data_lines[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    tool_delta = chunks[1]["choices"][0]["delta"]["tool_calls"][0]
    assert tool_delta["function"]["name"] == "read"
    assert json.loads(tool_delta["function"]["arguments"]) == {"path": "a.txt"}
    assert chunks[2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[3]["choices"] == []
    assert chunks[3]["usage"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_streaming_text_without_usage(api) -> None:
    client, fake = api
    fake.responses.append(ollama_response("hello"))
    response = await client.post(
        "/v1/chat/completions",
        json=chat_body(stream=True),
    )
    assert response.status_code == 200
    data_lines = [
        line[6:]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    chunks = [json.loads(line) for line in data_lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["content"] == "hello"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all("usage" not in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_title_generation_request_has_no_tools(api) -> None:
    client, fake = api
    fake.responses.append(ollama_response("Production 500 errors"))
    response = await client.post(
        "/v1/chat/completions",
        json=chat_body(tools=None),
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "Production 500 errors"
    )
    assert "tools" not in fake.requests[0]


@pytest.mark.asyncio
async def test_no_tools_ignores_upstream_tool_calls(api) -> None:
    client, fake = api
    fake.responses.append(
        ollama_response(
            "",
            tool_calls=[
                {"function": {"name": "read", "arguments": {"path": "x"}}}
            ],
        )
    )
    response = await client.post(
        "/v1/chat/completions", json=chat_body(tools=None)
    )
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == SAFE_NO_TOOLS
    assert "tool_calls" not in message


@pytest.mark.asyncio
async def test_unknown_tool_is_not_executed(api) -> None:
    client, fake = api
    fake.responses.append(
        ollama_response('{"name":"not_a_tool","arguments":{}}')
    )
    response = await client.post(
        "/v1/chat/completions", json=chat_body(tools=OPENCODE_TOOLS)
    )
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == SAFE_UNKNOWN
    assert "tool_calls" not in message


@pytest.mark.asyncio
async def test_security_violation_is_not_executed(api) -> None:
    client, fake = api
    fake.responses.append(
        ollama_response('{"name":"read","arguments":{"path":"..\\\\secret"}}')
    )
    response = await client.post(
        "/v1/chat/completions", json=chat_body(tools=OPENCODE_TOOLS)
    )
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == SAFE_SECURITY
    assert "tool_calls" not in message


@pytest.mark.asyncio
async def test_invalid_schema_is_not_executed(api) -> None:
    client, fake = api
    fake.responses.append(ollama_response('{"name":"read","arguments":{}}'))
    response = await client.post(
        "/v1/chat/completions", json=chat_body(tools=OPENCODE_TOOLS)
    )
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == SAFE_INVALID
    assert "tool_calls" not in message


@pytest.mark.asyncio
async def test_malformed_tool_call_is_not_leaked(api) -> None:
    client, fake = api
    fake.responses.append(
        ollama_response('<tool_call>{"name":"read","arguments":</tool_call>')
    )
    response = await client.post(
        "/v1/chat/completions", json=chat_body(tools=OPENCODE_TOOLS)
    )
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == SAFE_INVALID
    assert "arguments" not in message["content"]


@pytest.mark.asyncio
async def test_tool_choice_none_removes_tools(api) -> None:
    client, fake = api
    fake.responses.append(
        ollama_response('{"name":"read","arguments":{"path":"a.txt"}}')
    )
    response = await client.post(
        "/v1/chat/completions",
        json=chat_body(tools=OPENCODE_TOOLS, tool_choice="none"),
    )
    assert response.status_code == 200
    assert "tools" not in fake.requests[0]
    message = response.json()["choices"][0]["message"]
    assert message["content"] == SAFE_NO_TOOLS
    assert "tool_calls" not in message


@pytest.mark.asyncio
async def test_tool_result_ids_are_mapped_to_names(api) -> None:
    client, fake = api
    fake.responses.append(ollama_response("done"))
    messages = [
        {"role": "user", "content": "read a.txt"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"path":"a.txt"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "file data"},
    ]
    response = await client.post(
        "/v1/chat/completions",
        json=chat_body(messages=messages, tools=OPENCODE_TOOLS),
    )
    assert response.status_code == 200
    upstream_messages = fake.requests[0]["messages"]
    assert upstream_messages[1]["tool_calls"][0]["function"] == {
        "name": "read",
        "arguments": {"path": "a.txt"},
    }
    assert upstream_messages[2] == {
        "role": "tool",
        "content": "file data",
        "tool_name": "read",
    }


@pytest.mark.asyncio
async def test_cycle_limit_returns_safe_text(settings) -> None:
    settings = Settings(
        workspace=settings.workspace,
        allowed_models=settings.allowed_models,
        max_tool_cycles=1,
    )
    fake = FakeOllamaClient(
        [ollama_response('{"name":"read","arguments":{"path":"b.txt"}}')]
    )
    app = create_app(settings, ollama_client=fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        messages = [
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path":"a.txt"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "data"},
            {"role": "user", "content": "next"},
        ]
        response = await client.post(
            "/v1/chat/completions",
            json=chat_body(messages=messages, tools=OPENCODE_TOOLS),
        )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == SAFE_CYCLE


@pytest.mark.asyncio
async def test_repeat_limit_returns_safe_text(api) -> None:
    client, fake = api
    fake.responses.append(
        ollama_response('{"name":"read","arguments":{"path":"a.txt"}}')
    )
    messages = [
        {"role": "user", "content": "read a.txt"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"path":"a.txt"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "data"},
    ]
    response = await client.post(
        "/v1/chat/completions",
        json=chat_body(messages=messages, tools=OPENCODE_TOOLS),
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == SAFE_REPEAT


@pytest.mark.asyncio
async def test_model_not_allowed(api) -> None:
    client, fake = api
    response = await client.post(
        "/v1/chat/completions",
        json=chat_body(model="other:latest", tools=OPENCODE_TOOLS),
    )
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "model_not_allowed"
    assert fake.requests == []


@pytest.mark.asyncio
async def test_upstream_error_is_mapped(api) -> None:
    client, fake = api
    fake.responses.append(
        UpstreamError("model 'x' not found", status_code=404, code="model_not_found")
    )
    response = await client.post(
        "/v1/chat/completions", json=chat_body(tools=OPENCODE_TOOLS)
    )
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "model_not_found"


@pytest.mark.asyncio
async def test_empty_messages_rejected(api) -> None:
    client, _fake = api
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "qwen2.5-coder:7b", "messages": []},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_request_body_limit(settings) -> None:
    settings = Settings(
        workspace=settings.workspace,
        allowed_models=settings.allowed_models,
        max_request_bytes=1024,
    )
    fake = FakeOllamaClient()
    app = create_app(settings, ollama_client=fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=b"x" * 2000,
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


@pytest.mark.asyncio
async def test_models_and_health(api) -> None:
    client, fake = api
    models = await client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "qwen2.5-coder:7b"

    health = await client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["ollama"]["version"] == "0.40.11"

    fake.health_result = {"reachable": False}
    degraded = await client.get("/health")
    assert degraded.status_code == 200
    assert degraded.json()["status"] == "degraded"


@pytest.mark.asyncio
async def test_auth_required_for_v1_only(settings) -> None:
    settings = Settings(
        workspace=settings.workspace,
        allowed_models=settings.allowed_models,
        api_key="secret-token",
    )
    fake = FakeOllamaClient([ollama_response("hello")])
    app = create_app(settings, ollama_client=fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        health = await client.get("/health")
        assert health.status_code == 200

        unauthorized = await client.get("/v1/models")
        assert unauthorized.status_code == 401
        assert unauthorized.headers["www-authenticate"] == "Bearer"
        assert unauthorized.json()["error"]["code"] == "invalid_api_key"

        wrong = await client.get(
            "/v1/models", headers={"authorization": "Bearer wrong"}
        )
        assert wrong.status_code == 401

        authorized = await client.get(
            "/v1/models", headers={"authorization": "Bearer secret-token"}
        )
        assert authorized.status_code == 200

        chat = await client.post(
            "/v1/chat/completions",
            json=chat_body(tools=None),
            headers={"authorization": "Bearer secret-token"},
        )
        assert chat.status_code == 200
