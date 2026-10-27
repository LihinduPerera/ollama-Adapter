from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_PARTIAL_TAG_RE = re.compile(r"<tool_call>.*\Z", re.DOTALL | re.IGNORECASE)
_ORPHAN_CLOSE_RE = re.compile(r"</tool_call>", re.IGNORECASE)
_FENCED_JSON_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_TOOL_SHAPE_RE = re.compile(
    r'(?is)^\s*[\{\[].*(?:"arguments"|"parameters"|"tool_calls"|"function")\s*:'
)


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]
    source: str


@dataclass
class ParseResult:
    calls: list[ParsedToolCall] = field(default_factory=list)
    text: str = ""
    had_tool_syntax: bool = False
    errors: list[str] = field(default_factory=list)
    structured: bool = False


def content_to_text(content: Any) -> str:
    """Flatten OpenAI string-or-parts content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type")
                if part_type in {None, "text", "input_text", "output_text"}:
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif part_type == "image_url":
                    parts.append("[image omitted]")
        return "\n".join(parts)
    return str(content)


def parse_tool_calls(message: dict[str, Any], *, tools_advertised: bool) -> ParseResult:
    """Parse tool calls from a native Ollama assistant message.

    Structured ``message.tool_calls`` take precedence. When absent, the content
    is checked for ``<tool_call>`` blocks and for a bare JSON call. A bare JSON
    object is only considered a call when the *entire* content is JSON and it
    contains an ``arguments``/``parameters`` member.
    """
    content = content_to_text(message.get("content"))
    if not tools_advertised:
        return ParseResult(text=content)

    structured = message.get("tool_calls")
    if isinstance(structured, list) and structured:
        calls: list[ParsedToolCall] = []
        errors: list[str] = []
        for item in structured:
            item_calls, item_errors = _calls_from_payload(item, source="structured")
            calls.extend(item_calls)
            errors.extend(item_errors)
        return ParseResult(
            calls=calls,
            text=_clean_text(_TOOL_CALL_TAG_RE.sub("", content).strip()),
            had_tool_syntax=True,
            errors=errors,
            structured=True,
        )

    tags = list(_TOOL_CALL_TAG_RE.finditer(content))
    if tags:
        calls = []
        errors = []
        for match in tags:
            inner = match.group(1).strip()
            payloads = _load_json_values(inner)
            if payloads is None:
                errors.append("Invalid JSON inside a <tool_call> block")
                continue
            if not payloads:
                errors.append("Empty <tool_call> block")
                continue
            for payload in payloads:
                item_calls, item_errors = _calls_from_payload(
                    payload, source="tagged", require_arguments=False
                )
                calls.extend(item_calls)
                errors.extend(item_errors)
        text = _TOOL_CALL_TAG_RE.sub("", content).strip()
        return ParseResult(
            calls=calls,
            text=_clean_text(text),
            had_tool_syntax=True,
            errors=errors,
        )

    if _PARTIAL_TAG_RE.search(content) or _ORPHAN_CLOSE_RE.search(content):
        text = _PARTIAL_TAG_RE.sub("", content)
        text = _ORPHAN_CLOSE_RE.sub("", text)
        return ParseResult(
            calls=[],
            text=_clean_text(text),
            had_tool_syntax=True,
            errors=["Unclosed or malformed <tool_call> block"],
        )

    stripped = content.strip()
    if not stripped:
        return ParseResult(text="")

    fenced = _FENCED_JSON_RE.match(stripped)
    candidate = fenced.group(1).strip() if fenced else stripped
    if candidate.startswith(("{", "[")):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            if _TOOL_SHAPE_RE.match(candidate):
                return ParseResult(
                    calls=[],
                    text="",
                    had_tool_syntax=True,
                    errors=["Invalid JSON in a bare tool call"],
                )
            return ParseResult(text=content.strip())

        calls, errors = _calls_from_payload(
            payload, source="bare_json", require_arguments=True
        )
        if _looks_like_call_payload(payload) and (calls or errors):
            return ParseResult(
                calls=calls,
                text="",
                had_tool_syntax=True,
                errors=errors,
            )

    return ParseResult(text=content.strip())


def _looks_like_call_payload(payload: Any) -> bool:
    if isinstance(payload, dict):
        return any(
            key in payload
            for key in (
                "arguments",
                "parameters",
                "function",
                "tool_calls",
                "tool",
                "tool_name",
            )
        )
    if isinstance(payload, list):
        return bool(payload) and all(
            isinstance(item, dict) and _looks_like_call_payload(item)
            for item in payload
        )
    return False


def _load_json_values(text: str) -> list[Any] | None:
    """Decode one or more JSON values from a tag body, separated by whitespace."""
    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        if text[index] == ",":
            index += 1
            continue
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return None
        values.append(value)
        index = end
    return values


def _calls_from_payload(
    payload: Any,
    *,
    source: str,
    require_arguments: bool = True,
) -> tuple[list[ParsedToolCall], list[str]]:
    if isinstance(payload, list):
        calls: list[ParsedToolCall] = []
        errors: list[str] = []
        for item in payload:
            item_calls, item_errors = _calls_from_payload(
                item, source=source, require_arguments=require_arguments
            )
            calls.extend(item_calls)
            errors.extend(item_errors)
        return calls, errors

    if not isinstance(payload, dict):
        return [], ["Tool call must be a JSON object"]

    if "tool_calls" in payload:
        inner = payload.get("tool_calls")
        if not isinstance(inner, list):
            return [], ["tool_calls must be a JSON array"]
        return _calls_from_payload(
            inner, source=source, require_arguments=require_arguments
        )

    function = payload.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = payload.get("name", payload.get("tool_name", payload.get("tool")))
        if "arguments" in payload:
            arguments = payload.get("arguments")
        elif "parameters" in payload:
            arguments = payload.get("parameters")
        elif require_arguments:
            return [], ["Tool call is missing an arguments object"]
        else:
            arguments = {}

    if not isinstance(name, str) or not name.strip():
        return [], ["Tool call is missing a tool name"]

    normalized_args, error = _normalize_arguments(arguments)
    if error:
        return [], [error]
    return [ParsedToolCall(name=name.strip(), arguments=normalized_args, source=source)], []


def _normalize_arguments(arguments: Any) -> tuple[dict[str, Any], str | None]:
    if arguments is None:
        return {}, None
    if isinstance(arguments, str):
        value = arguments.strip()
        if not value:
            return {}, None
        try:
            arguments = json.loads(value)
        except json.JSONDecodeError:
            return {}, "Tool call arguments are not valid JSON"
    if not isinstance(arguments, dict):
        return {}, "Tool call arguments must be a JSON object"
    return arguments, None


def _clean_text(text: str) -> str:
    """Remove a bare JSON tool call that leaked into assistant text."""
    stripped = text.strip()
    if not stripped:
        return ""
    candidate = stripped
    fenced = _FENCED_JSON_RE.match(stripped)
    if fenced:
        candidate = fenced.group(1).strip()
    if not candidate.startswith(("{", "[")):
        return stripped
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return stripped
    calls, _ = _calls_from_payload(payload, source="text", require_arguments=True)
    return "" if calls else stripped
