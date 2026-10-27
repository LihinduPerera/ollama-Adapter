from __future__ import annotations

from adapter.tool_parser import content_to_text, parse_tool_calls


def test_structured_tool_call_with_object_arguments() -> None:
    result = parse_tool_calls(
        {
            "content": "",
            "tool_calls": [
                {"function": {"name": "read", "arguments": {"path": "a.txt"}}}
            ],
        },
        tools_advertised=True,
    )
    assert result.structured is True
    assert result.had_tool_syntax is True
    assert result.errors == []
    assert len(result.calls) == 1
    assert result.calls[0].name == "read"
    assert result.calls[0].arguments == {"path": "a.txt"}


def test_structured_tool_call_with_json_string_arguments() -> None:
    result = parse_tool_calls(
        {
            "content": "I will read it.",
            "tool_calls": [
                {
                    "function": {
                        "name": "read",
                        "arguments": '{"path":"a.txt","limit":5}',
                    }
                }
            ],
        },
        tools_advertised=True,
    )
    assert result.calls[0].arguments == {"path": "a.txt", "limit": 5}
    assert result.text == "I will read it."


def test_bare_json_tool_call() -> None:
    result = parse_tool_calls(
        {"content": '{"name":"get_directory","arguments":{"path":"."}}'},
        tools_advertised=True,
    )
    assert result.had_tool_syntax is True
    assert result.errors == []
    assert result.calls[0].name == "get_directory"
    assert result.calls[0].arguments == {"path": "."}
    assert result.text == ""


def test_bare_json_function_wrapper() -> None:
    result = parse_tool_calls(
        {
            "content": (
                '{"function":{"name":"read_file",'
                '"arguments":{"file_path":"README.md"}}}'
            )
        },
        tools_advertised=True,
    )
    assert result.calls[0].name == "read_file"
    assert result.calls[0].arguments == {"file_path": "README.md"}


def test_bare_json_array_of_calls() -> None:
    result = parse_tool_calls(
        {
            "content": (
                '[{"name":"read","arguments":{"path":"a.txt"}},'
                '{"name":"read","arguments":{"path":"b.txt"}}]'
            )
        },
        tools_advertised=True,
    )
    assert len(result.calls) == 2
    assert [call.arguments["path"] for call in result.calls] == ["a.txt", "b.txt"]


def test_tagged_tool_call_with_surrounding_text() -> None:
    result = parse_tool_calls(
        {
            "content": (
                "Let me look.\n"
                '<tool_call>{"name":"read","arguments":{"path":"a.txt"}}</tool_call>\n'
                "Then I will continue."
            )
        },
        tools_advertised=True,
    )
    assert len(result.calls) == 1
    assert result.calls[0].name == "read"
    assert result.text == "Let me look.\n\nThen I will continue."


def test_multiple_tagged_calls() -> None:
    result = parse_tool_calls(
        {
            "content": (
                '<tool_call>{"name":"read","arguments":{"path":"a.txt"}}</tool_call>'
                '<tool_call>{"name":"read","arguments":{"path":"b.txt"}}</tool_call>'
            )
        },
        tools_advertised=True,
    )
    assert len(result.calls) == 2
    assert result.text == ""


def test_tagged_call_with_missing_arguments_defaults_to_empty_object() -> None:
    result = parse_tool_calls(
        {"content": '<tool_call>{"name":"get_directory"}</tool_call>'},
        tools_advertised=True,
    )
    assert result.calls[0].arguments == {}


def test_fenced_json_tool_call() -> None:
    result = parse_tool_calls(
        {
            "content": (
                "```json\n"
                '{"name":"read","arguments":{"path":"a.txt"}}\n'
                "```"
            )
        },
        tools_advertised=True,
    )
    assert len(result.calls) == 1
    assert result.text == ""


def test_plain_text_is_not_a_tool_call() -> None:
    result = parse_tool_calls(
        {"content": "The answer is 42."}, tools_advertised=True
    )
    assert result.calls == []
    assert result.errors == []
    assert result.had_tool_syntax is False
    assert result.text == "The answer is 42."


def test_plain_json_without_arguments_is_not_a_tool_call() -> None:
    result = parse_tool_calls(
        {"content": '{"name":"Alice","age":30}'}, tools_advertised=True
    )
    assert result.calls == []
    assert result.had_tool_syntax is False
    assert result.text == '{"name":"Alice","age":30}'


def test_malformed_tag_reports_error_without_leaking_content() -> None:
    result = parse_tool_calls(
        {"content": '<tool_call>{"name":"read","arguments":</tool_call>'},
        tools_advertised=True,
    )
    assert result.calls == []
    assert result.had_tool_syntax is True
    assert result.text == ""
    assert result.errors


def test_unclosed_tag_reports_error() -> None:
    result = parse_tool_calls(
        {"content": 'Before <tool_call>{"name":"read"}'}, tools_advertised=True
    )
    assert result.calls == []
    assert result.had_tool_syntax is True
    assert result.text == "Before"
    assert result.errors


def test_empty_tag_reports_error() -> None:
    result = parse_tool_calls(
        {"content": "<tool_call>   </tool_call>"}, tools_advertised=True
    )
    assert result.calls == []
    assert result.had_tool_syntax is True
    assert result.errors


def test_invalid_bare_json_tool_shape_reports_error() -> None:
    result = parse_tool_calls(
        {"content": '{"name":"read","arguments":{"path":"a.txt"'},
        tools_advertised=True,
    )
    assert result.calls == []
    assert result.had_tool_syntax is True
    assert result.text == ""
    assert result.errors


def test_tools_not_advertised_never_parses_calls() -> None:
    result = parse_tool_calls(
        {"content": '<tool_call>{"name":"read","arguments":{"path":"a.txt"}}</tool_call>'},
        tools_advertised=False,
    )
    assert result.calls == []
    assert result.had_tool_syntax is False
    assert result.text.startswith("<tool_call>")


def test_structured_calls_take_precedence_over_content_tags() -> None:
    result = parse_tool_calls(
        {
            "content": '<tool_call>{"name":"write","arguments":{}}</tool_call>',
            "tool_calls": [
                {"function": {"name": "read", "arguments": {"path": "a.txt"}}}
            ],
        },
        tools_advertised=True,
    )
    assert len(result.calls) == 1
    assert result.calls[0].name == "read"
    assert result.text == ""


def test_content_to_text_flattens_parts() -> None:
    assert (
        content_to_text(
            [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                {"type": "text", "text": "world"},
            ]
        )
        == "hello\n[image omitted]\nworld"
    )
