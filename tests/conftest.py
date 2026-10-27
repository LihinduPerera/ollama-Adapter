from __future__ import annotations

from typing import Any

import pytest

from adapter.config import Settings
from adapter.ollama_client import OllamaClient


class FakeOllamaClient:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.requests: list[dict[str, Any]] = []
        self.closed = False
        self.health_result: dict[str, Any] = {
            "reachable": True,
            "version": "0.40.11",
        }

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(payload)
        if not self.responses:
            raise AssertionError("FakeOllamaClient has no queued response")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def close(self) -> None:
        self.closed = True

    async def health(self, *, timeout: float = 2.0) -> dict[str, Any]:
        return dict(self.health_result)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        workspace=tmp_path,
        allowed_models=("qwen2.5-coder:7b",),
        api_key=None,
        max_tool_calls=4,
        max_tool_cycles=12,
        repeat_call_limit=2,
    )


@pytest.fixture
def fake_client() -> FakeOllamaClient:
    return FakeOllamaClient()


def ollama_response(
    content: str | None = None,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    done_reason: str = "stop",
    prompt_eval_count: int = 10,
    eval_count: int = 5,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "model": "qwen2.5-coder:7b",
        "created_at": "2026-10-27T00:00:00Z",
        "message": message,
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }


def tool_definition(
    name: str,
    parameters: dict[str, Any] | None = None,
    description: str = "",
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
            or {"type": "object", "properties": {}},
        },
    }


OPENCODE_TOOLS = [
    tool_definition(
        "read",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 0},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    tool_definition(
        "write",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    ),
    tool_definition(
        "edit",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "oldString": {"type": "string"},
                "newString": {"type": "string"},
                "replaceAll": {"type": "boolean"},
            },
            "required": ["path", "oldString", "newString"],
            "additionalProperties": False,
        },
    ),
    tool_definition(
        "shell",
        {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "workdir": {"type": "string"},
                "timeout": {"type": "number"},
                "background": {"type": "boolean"},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    ),
    tool_definition(
        "glob",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    ),
    tool_definition(
        "grep",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "include": {"type": "string"},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    ),
]
