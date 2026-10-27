from __future__ import annotations

import pytest

from adapter.config import Settings
from adapter.errors import SecurityError, ToolValidationError
from adapter.protocol import (
    SAFE_CYCLE,
    SAFE_INVALID,
    SAFE_LIMIT,
    SAFE_REPEAT,
    SAFE_SECURITY,
    SAFE_UNKNOWN,
    ChatCompletionRequest,
    ChatMessage,
    OpenAIToolCall,
    ToolCallFunction,
    TOOL_OUTPUT_TRUNCATED_SUFFIX,
    ToolDefinition,
    ToolRegistry,
    build_ollama_messages,
    build_ollama_payload,
    check_repeated_calls,
    count_tool_cycles,
    filter_tools,
    finish_reason_from_ollama,
    make_completion,
    usage_from_ollama,
    validate_calls,
)
from adapter.security import WorkspaceGuard
from adapter.tool_parser import ParsedToolCall
from conftest import OPENCODE_TOOLS, tool_definition


def make_registry(tools=None) -> ToolRegistry:
    definitions = tools if tools is not None else OPENCODE_TOOLS
    return ToolRegistry([ToolDefinition.model_validate(item) for item in definitions])


def test_alias_only_when_target_is_advertised() -> None:
    registry = make_registry([tool_definition("read")])
    assert registry.resolve_name("get_directory") == "read"
    assert registry.resolve_name("read_file") == "read"
    assert registry.resolve_name("write_file") is None


def test_exact_tool_name_wins_over_alias() -> None:
    registry = make_registry(
        [tool_definition("get_directory"), tool_definition("read")]
    )
    assert registry.resolve_name("get_directory") == "get_directory"


def test_case_insensitive_resolution() -> None:
    registry = make_registry([tool_definition("read")])
    assert registry.resolve_name("READ") == "read"


def test_read_argument_aliases_and_coercion() -> None:
    registry = make_registry()
    args = registry.normalize_arguments(
        "read",
        {"file_path": "a.txt", "start_line": "2", "max_lines": "5"},
    )
    assert args == {"path": "a.txt", "offset": 2, "limit": 5}


def test_directory_alias_defaults_path() -> None:
    registry = make_registry()
    args = registry.normalize_arguments("read", {}, original_name="get_directory")
    assert args["path"] == "."


def test_read_file_alias_does_not_default_path() -> None:
    registry = make_registry()
    args = registry.normalize_arguments("read", {}, original_name="read_file")
    assert "path" not in args


def test_write_argument_aliases() -> None:
    registry = make_registry()
    args = registry.normalize_arguments(
        "write", {"file_path": "a.txt", "text": "hello"}
    )
    assert args == {"path": "a.txt", "content": "hello"}


def test_edit_argument_aliases() -> None:
    registry = make_registry()
    args = registry.normalize_arguments(
        "edit",
        {
            "file_path": "a.txt",
            "old_string": "a",
            "new_string": "b",
            "replace_all": "true",
        },
    )
    assert args == {
        "path": "a.txt",
        "oldString": "a",
        "newString": "b",
        "replaceAll": True,
    }


def test_shell_argument_aliases() -> None:
    registry = make_registry()
    args = registry.normalize_arguments(
        "shell",
        {"cmd": "Get-ChildItem", "cwd": ".", "timeout_ms": "5000"},
    )
    assert args["command"] == "Get-ChildItem"
    assert args["workdir"] == "."
    assert args["timeout"] == pytest.approx(5000.0)


def test_glob_and_grep_defaults_and_aliases() -> None:
    registry = make_registry()
    assert registry.normalize_arguments("glob", {"glob": "**/*.py"}) == {
        "pattern": "**/*.py",
        "path": ".",
    }
    assert registry.normalize_arguments(
        "grep", {"regex": "foo", "glob": "*.py"}
    ) == {"pattern": "foo", "include": "*.py", "path": "."}


def test_schema_requires_required_fields() -> None:
    registry = make_registry()
    with pytest.raises(ToolValidationError):
        registry.validate("read", {})


def test_schema_rejects_wrong_types() -> None:
    registry = make_registry()
    with pytest.raises(ToolValidationError):
        registry.validate("read", {"path": 123})


def test_unknown_top_level_arguments_are_dropped() -> None:
    registry = make_registry()
    args = registry.validate("read", {"path": "a.txt", "unexpected": "x"})
    assert args == {"path": "a.txt"}


def test_unknown_tool_is_rejected() -> None:
    registry = make_registry()
    with pytest.raises(ToolValidationError):
        registry.validate("not_a_tool", {})


def test_duplicate_tool_names_are_rejected() -> None:
    with pytest.raises(ToolValidationError):
        make_registry([tool_definition("read"), tool_definition("read")])


def test_invalid_advertised_schema_is_rejected() -> None:
    registry = make_registry(
        [
            tool_definition(
                "bad",
                {
                    "type": "object",
                    "properties": {"x": {"type": "not-a-real-type"}},
                },
            )
        ]
    )
    with pytest.raises(ToolValidationError):
        registry.validate("bad", {"x": "value"})


def test_filter_tools_none_and_specific() -> None:
    tools = [ToolDefinition.model_validate(item) for item in OPENCODE_TOOLS]
    assert filter_tools(tools, "none") == []
    selected = filter_tools(
        tools, {"type": "function", "function": {"name": "read"}}
    )
    assert [tool.function.name for tool in selected] == ["read"]
    with pytest.raises(ToolValidationError):
        filter_tools(tools, {"type": "function", "function": {"name": "missing"}})


def test_build_ollama_messages_maps_tool_call_ids_to_names() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                OpenAIToolCall(
                    id="call_1",
                    function=ToolCallFunction(
                        name="read", arguments='{"path":"a.txt"}'
                    ),
                )
            ],
        ),
        ChatMessage(role="tool", content="file data", tool_call_id="call_1"),
    ]
    converted = build_ollama_messages(messages)
    assert converted[0]["tool_calls"][0]["function"] == {
        "name": "read",
        "arguments": {"path": "a.txt"},
    }
    assert converted[1] == {
        "role": "tool",
        "content": "file data",
        "tool_name": "read",
    }


def test_build_ollama_messages_falls_back_to_message_name() -> None:
    messages = [
        ChatMessage(
            role="tool",
            content="data",
            tool_call_id="unknown",
            name="read",
        )
    ]
    assert build_ollama_messages(messages)[0]["tool_name"] == "read"


def test_build_ollama_messages_truncates_tool_output() -> None:
    messages = [
        ChatMessage(role="tool", content="x" * 120, name="shell"),
    ]
    converted = build_ollama_messages(messages, max_tool_result_chars=100)
    assert converted[0]["content"] == "x" * 100 + TOOL_OUTPUT_TRUNCATED_SUFFIX
    assert converted[0]["tool_name"] == "shell"

    short = build_ollama_messages(
        [ChatMessage(role="tool", content="short", name="shell")],
        max_tool_result_chars=100,
    )
    assert short[0]["content"] == "short"


def test_build_ollama_payload_options_and_tools(settings) -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "ollama-adapter/qwen2.5-coder:7b",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": OPENCODE_TOOLS,
            "stream": True,
            "max_completion_tokens": 8192,
        }
    )
    registry = make_registry()
    payload = build_ollama_payload(
        request, settings, list(registry.specs.values())
    )
    assert payload["model"] == "qwen2.5-coder:7b"
    assert payload["stream"] is False
    assert payload["options"]["num_ctx"] == 16384
    assert payload["options"]["num_predict"] == 4096
    assert len(payload["tools"]) == len(OPENCODE_TOOLS)
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


def test_usage_and_finish_reason() -> None:
    usage = usage_from_ollama({"prompt_eval_count": 10, "eval_count": 5})
    assert usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert (
        finish_reason_from_ollama({"done_reason": "length"}, has_tools=False)
        == "length"
    )
    assert (
        finish_reason_from_ollama({"done_reason": "stop"}, has_tools=True)
        == "tool_calls"
    )


def test_completion_sse_chunks_contain_valid_tool_call() -> None:
    from adapter.protocol import ValidatedToolCall

    completion = make_completion(
        model="qwen2.5-coder:7b",
        content=None,
        tool_calls=[
            ValidatedToolCall(
                id="call_1", name="read", arguments={"path": "a.txt"}
            )
        ],
        finish_reason="tool_calls",
        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    )
    chunks = completion.to_sse_chunks(include_usage=True)
    assert len(chunks) == 4
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    tool_delta = chunks[1]["choices"][0]["delta"]["tool_calls"][0]
    assert tool_delta["id"] == "call_1"
    assert tool_delta["function"]["name"] == "read"
    assert tool_delta["function"]["arguments"] == '{"path":"a.txt"}'
    assert chunks[2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[3]["choices"] == []
    assert chunks[3]["usage"]["total_tokens"] == 3


def test_repeat_detection_blocks_immediate_repeat() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                OpenAIToolCall(
                    id="call_1",
                    function=ToolCallFunction(
                        name="read", arguments='{"path":"a.txt"}'
                    ),
                )
            ],
        ),
        ChatMessage(role="tool", content="data", tool_call_id="call_1"),
    ]
    calls = [ParsedToolCall(name="read", arguments={"path": "a.txt"}, source="bare")]
    assert check_repeated_calls(calls, messages, repeat_limit=2) == SAFE_REPEAT


def test_repeat_detection_allows_repeat_after_user_message() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                OpenAIToolCall(
                    id="call_1",
                    function=ToolCallFunction(
                        name="read", arguments='{"path":"a.txt"}'
                    ),
                )
            ],
        ),
        ChatMessage(role="tool", content="data", tool_call_id="call_1"),
        ChatMessage(role="user", content="read it again"),
    ]
    calls = [ParsedToolCall(name="read", arguments={"path": "a.txt"}, source="bare")]
    assert check_repeated_calls(calls, messages, repeat_limit=2) is None


def test_repeat_detection_blocks_third_occurrence() -> None:
    messages = []
    for index in range(2):
        messages.extend(
            [
                ChatMessage(
                    role="assistant",
                    tool_calls=[
                        OpenAIToolCall(
                            id=f"call_{index}",
                            function=ToolCallFunction(
                                name="read", arguments='{"path":"a.txt"}'
                            ),
                        )
                    ],
                ),
                ChatMessage(
                    role="tool", content="data", tool_call_id=f"call_{index}"
                ),
                ChatMessage(role="user", content="again"),
            ]
        )
    calls = [ParsedToolCall(name="read", arguments={"path": "a.txt"}, source="bare")]
    assert check_repeated_calls(calls, messages, repeat_limit=2) == SAFE_REPEAT


def test_count_tool_cycles() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                OpenAIToolCall(
                    id="call_1",
                    function=ToolCallFunction(name="read", arguments="{}"),
                )
            ],
        )
    ]
    assert count_tool_cycles(messages) == 1


def test_validate_calls_resolves_alias_and_normalizes() -> None:
    settings = Settings(workspace=__import__("pathlib").Path.cwd())
    registry = make_registry()
    guard = WorkspaceGuard(settings.workspace)
    calls = [
        ParsedToolCall(
            name="get_directory", arguments={"path": "."}, source="bare"
        )
    ]
    validated, error = validate_calls(
        calls,
        registry=registry,
        guard=guard,
        settings=settings,
        messages=[ChatMessage(role="user", content="list")],
    )
    assert error is None
    assert validated[0].name == "read"
    assert validated[0].arguments == {"path": "."}


def test_validate_calls_unknown_tool() -> None:
    settings = Settings(workspace=__import__("pathlib").Path.cwd())
    registry = make_registry()
    guard = WorkspaceGuard(settings.workspace)
    validated, error = validate_calls(
        [ParsedToolCall(name="unknown", arguments={}, source="bare")],
        registry=registry,
        guard=guard,
        settings=settings,
        messages=[ChatMessage(role="user", content="x")],
    )
    assert validated == []
    assert error == SAFE_UNKNOWN


def test_validate_calls_security_rejection(tmp_path) -> None:
    settings = Settings(workspace=tmp_path)
    registry = make_registry()
    guard = WorkspaceGuard(tmp_path)
    validated, error = validate_calls(
        [ParsedToolCall(name="read", arguments={"path": "..\\secret"}, source="bare")],
        registry=registry,
        guard=guard,
        settings=settings,
        messages=[ChatMessage(role="user", content="x")],
    )
    assert validated == []
    assert error == SAFE_SECURITY


def test_validate_calls_schema_rejection(tmp_path) -> None:
    settings = Settings(workspace=tmp_path)
    registry = make_registry()
    guard = WorkspaceGuard(tmp_path)
    validated, error = validate_calls(
        [ParsedToolCall(name="read", arguments={}, source="bare")],
        registry=registry,
        guard=guard,
        settings=settings,
        messages=[ChatMessage(role="user", content="x")],
    )
    assert validated == []
    assert error == SAFE_INVALID


def test_validate_calls_limit_and_cycle(tmp_path) -> None:
    settings = Settings(workspace=tmp_path, max_tool_calls=1)
    registry = make_registry()
    guard = WorkspaceGuard(tmp_path)
    calls = [
        ParsedToolCall(name="read", arguments={"path": "a"}, source="bare"),
        ParsedToolCall(name="read", arguments={"path": "b"}, source="bare"),
    ]
    validated, error = validate_calls(
        calls,
        registry=registry,
        guard=guard,
        settings=settings,
        messages=[ChatMessage(role="user", content="x")],
    )
    assert validated == []
    assert error == SAFE_LIMIT

    settings = Settings(workspace=tmp_path, max_tool_cycles=1)
    history = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                OpenAIToolCall(
                    id="call_1",
                    function=ToolCallFunction(name="read", arguments='{"path":"a"}'),
                )
            ],
        )
    ]
    validated, error = validate_calls(
        [ParsedToolCall(name="read", arguments={"path": "b"}, source="bare")],
        registry=registry,
        guard=guard,
        settings=settings,
        messages=history,
    )
    assert validated == []
    assert error == SAFE_CYCLE
