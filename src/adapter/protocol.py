from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import jsonschema
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .errors import SecurityError, ToolValidationError
from .security import WorkspaceGuard
from .tool_parser import ParsedToolCall, content_to_text

logger = logging.getLogger(__name__)

SAFE_INVALID = (
    "The model produced a tool call that the adapter could not validate. "
    "No tool was executed."
)
SAFE_SECURITY = (
    "The model requested a tool call that violates the adapter security policy. "
    "No tool was executed."
)
SAFE_LIMIT = (
    "The model exceeded the adapter's tool-call limit. No tool was executed."
)
SAFE_REPEAT = (
    "The model repeated a tool call that already ran. "
    "No duplicate tool call was executed."
)
SAFE_CYCLE = (
    "The model exceeded the adapter's tool-call cycle limit. No tool was executed."
)
SAFE_UNKNOWN = (
    "The model requested a tool that is not available in this request. "
    "No tool was executed."
)
SAFE_NO_TOOLS = (
    "The model attempted a tool call, but no tools were available in this request."
)


class ToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    arguments: Any = None


class OpenAIToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    type: str = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: Any = None
    name: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None
    tool_call_id: str | None = None


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    strict: bool | None = None


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "function"
    function: FunctionDefinition


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    tools: list[ToolDefinition] | None = None
    tool_choice: Any = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: str | list[str] | None = None
    seed: int | None = None
    user: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool = False

    def to_ollama(self) -> dict[str, Any]:
        function: dict[str, Any] = {
            "name": self.name,
            "parameters": self.parameters,
        }
        if self.description:
            function["description"] = self.description
        return {"type": "function", "function": function}


# Common names emitted by text-based tool-call models. Aliases are only used
# when the target tool is actually advertised in the current request.
_ALIASES: dict[str, str] = {
    "get_directory": "read",
    "list_directory": "read",
    "list_dir": "read",
    "list_files": "read",
    "ls": "read",
    "dir": "read",
    "read_file": "read",
    "readfile": "read",
    "cat": "read",
    "view_file": "read",
    "open_file": "read",
    "write_file": "write",
    "writefile": "write",
    "create_file": "write",
    "save_file": "write",
    "edit_file": "edit",
    "editfile": "edit",
    "replace_in_file": "edit",
    "str_replace": "edit",
    "run_command": "shell",
    "execute_command": "shell",
    "run_shell": "shell",
    "bash": "shell",
    "sh": "shell",
    "powershell": "shell",
    "terminal": "shell",
    "glob_files": "glob",
    "find_files": "glob",
    "search_files": "grep",
    "ripgrep": "grep",
}

_DIRECTORY_ALIASES = frozenset(
    {"get_directory", "list_directory", "list_dir", "list_files", "ls", "dir"}
)

_COMMON_PATH_ALIASES = {
    "file_path": "path",
    "filepath": "path",
    "filePath": "path",
    "file_name": "path",
    "filename": "path",
    "file": "path",
    "directory": "path",
    "dir": "path",
    "folder": "path",
}

_ARGUMENT_ALIASES: dict[str, dict[str, str]] = {
    "read": {
        **_COMMON_PATH_ALIASES,
        "start_line": "offset",
        "line_offset": "offset",
        "max_lines": "limit",
        "num_lines": "limit",
    },
    "write": {
        **_COMMON_PATH_ALIASES,
        "text": "content",
        "contents": "content",
        "file_content": "content",
        "fileContent": "content",
        "data": "content",
    },
    "edit": {
        **_COMMON_PATH_ALIASES,
        "old_string": "oldString",
        "old_text": "oldString",
        "oldText": "oldString",
        "old": "oldString",
        "new_string": "newString",
        "new_text": "newString",
        "newText": "newString",
        "new": "newString",
        "replace_all": "replaceAll",
    },
    "shell": {
        "cmd": "command",
        "command_line": "command",
        "commandLine": "command",
        "script": "command",
        "shell_command": "command",
        "cwd": "workdir",
        "working_directory": "workdir",
        "working_dir": "workdir",
        "work_dir": "workdir",
        "directory": "workdir",
        "dir": "workdir",
        "timeout_ms": "timeout",
        "timeoutMs": "timeout",
    },
    "glob": {
        "file_pattern": "pattern",
        "filePattern": "pattern",
        "glob": "pattern",
        "glob_pattern": "pattern",
        "root": "path",
        "dir": "path",
        "directory": "path",
    },
    "grep": {
        "regex": "pattern",
        "query": "pattern",
        "search": "pattern",
        "text": "pattern",
        "file_pattern": "include",
        "filePattern": "include",
        "glob": "include",
        "include_pattern": "include",
        "root": "path",
        "dir": "path",
        "directory": "path",
    },
}

_JSON_SCHEMA_VALIDATOR_CACHE: dict[str, Any] = {}


class ToolRegistry:
    def __init__(self, tools: list[ToolDefinition] | None) -> None:
        self.specs: dict[str, ToolSpec] = {}
        self._lower_names: dict[str, str] = {}
        self._aliases: dict[str, str] = {}

        for tool in tools or []:
            if tool.type != "function":
                raise ToolValidationError(
                    f"Unsupported tool type {tool.type!r}; only function tools are supported"
                )
            function = tool.function
            name = function.name.strip()
            if not name:
                raise ToolValidationError("A tool has an empty name")
            if len(name) > 128:
                raise ToolValidationError("A tool name is longer than 128 characters")
            if name in self.specs:
                raise ToolValidationError(f"Duplicate tool name in request: {name}")
            self.specs[name] = ToolSpec(
                name=name,
                description=function.description or "",
                parameters=function.parameters,
                strict=bool(function.strict),
            )

        for name in self.specs:
            self._lower_names.setdefault(name.lower(), name)
        for alias, target in _ALIASES.items():
            if alias in self.specs:
                continue
            if target in self.specs:
                self._aliases[alias.lower()] = target

    @property
    def advertised(self) -> bool:
        return bool(self.specs)

    @property
    def names(self) -> list[str]:
        return list(self.specs)

    def resolve_name(self, name: str) -> str | None:
        if name in self.specs:
            return name
        lowered = name.lower()
        if lowered in self._lower_names:
            return self._lower_names[lowered]
        return self._aliases.get(lowered)

    def normalize_arguments(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        original_name: str | None = None,
    ) -> dict[str, Any]:
        spec = self.specs.get(name)
        if spec is None:
            raise ToolValidationError("Tool is not advertised in this request")
        args = dict(arguments)

        for alias, target in _ARGUMENT_ALIASES.get(name, {}).items():
            if alias == target:
                continue
            if alias in args:
                if target not in args:
                    args[target] = args[alias]
                args.pop(alias, None)

        if (
            name == "read"
            and original_name is not None
            and original_name.lower() in _DIRECTORY_ALIASES
            and "path" not in args
        ):
            args["path"] = "."
        if name in {"glob", "grep"} and "path" not in args:
            args["path"] = "."

        schema = spec.parameters
        properties = schema.get("properties")
        required = set(schema.get("required") or [])
        if isinstance(properties, dict):
            if schema.get("additionalProperties") is False:
                args = {key: value for key, value in args.items() if key in properties}
            for key in list(args):
                value = args[key]
                if value is None and key not in required:
                    prop = properties.get(key, {})
                    types = prop.get("type")
                    if not (
                        types == "null"
                        or (isinstance(types, list) and "null" in types)
                    ):
                        args.pop(key)
                        continue
                prop = properties.get(key)
                if isinstance(prop, dict):
                    args[key] = _coerce_value(value, prop)

        return args

    def validate_schema(self, name: str, arguments: dict[str, Any]) -> None:
        spec = self.specs.get(name)
        if spec is None:
            raise ToolValidationError("Tool is not advertised in this request")
        validator = self._validator(spec)
        errors = sorted(validator.iter_errors(arguments), key=lambda error: list(error.path))
        if not errors:
            return
        details: list[str] = []
        for error in errors[:3]:
            path = ".".join(str(part) for part in error.path) or "<root>"
            details.append(f"{path}: {error.validator} constraint failed")
        raise ToolValidationError(
            "Tool arguments do not match the advertised schema: " + "; ".join(details)
        )

    def validate(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        original_name: str | None = None,
    ) -> dict[str, Any]:
        normalized = self.normalize_arguments(
            name, arguments, original_name=original_name
        )
        self.validate_schema(name, normalized)
        return normalized

    def _validator(self, spec: ToolSpec) -> Any:
        schema_json = json.dumps(
            spec.parameters,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        cache_key = (
            spec.name
            + ":"
            + hashlib.sha256(schema_json.encode("utf-8")).hexdigest()
        )
        cached = _JSON_SCHEMA_VALIDATOR_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            validator_class = jsonschema.validators.validator_for(spec.parameters)
            validator_class.check_schema(spec.parameters)
            validator = validator_class(spec.parameters)
        except jsonschema.SchemaError as exc:
            raise ToolValidationError(
                "The advertised JSON Schema for this tool is invalid"
            ) from exc
        _JSON_SCHEMA_VALIDATOR_CACHE[cache_key] = validator
        return validator


def _coerce_value(value: Any, schema: dict[str, Any]) -> Any:
    types = schema.get("type")
    if isinstance(types, str):
        types = [types]
    if not isinstance(types, list):
        return value
    if value is None:
        return value
    if isinstance(value, str):
        text = value.strip()
        if "integer" in types and re.fullmatch(r"-?\d+", text):
            return int(text)
        if "number" in types:
            try:
                return float(text)
            except ValueError:
                pass
        if "boolean" in types and text.lower() in {"true", "false"}:
            return text.lower() == "true"
        if "object" in types and text.startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return value
            if isinstance(parsed, dict):
                return parsed
        if "array" in types and text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return value
            if isinstance(parsed, list):
                return parsed
    return value


def filter_tools(
    tools: list[ToolDefinition] | None, tool_choice: Any
) -> list[ToolDefinition]:
    if not tools:
        return []
    if tool_choice == "none":
        return []
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not name:
            raise ToolValidationError("tool_choice.function.name is required")
        selected = [tool for tool in tools if tool.function.name == name]
        if not selected:
            raise ToolValidationError("tool_choice references a tool that is not advertised")
        return selected
    return tools


TOOL_OUTPUT_TRUNCATED_SUFFIX = "\n\n[adapter: tool output truncated]"


def build_ollama_messages(
    messages: list[ChatMessage], *, max_tool_result_chars: int | None = None
) -> list[dict[str, Any]]:
    id_to_name: dict[str, str] = {}
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            for call in message.tool_calls:
                if call.id:
                    id_to_name[call.id] = call.function.name

    converted: list[dict[str, Any]] = []
    for message in messages:
        role = "system" if message.role == "developer" else message.role
        content = content_to_text(message.content)
        images = _extract_images(message.content)

        if role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": content}
            if message.tool_calls:
                calls: list[dict[str, Any]] = []
                for call in message.tool_calls:
                    arguments = _arguments_dict(call.function.arguments)
                    calls.append(
                        {
                            "function": {
                                "name": call.function.name,
                                "arguments": arguments,
                            }
                        }
                    )
                if calls:
                    item["tool_calls"] = calls
            if images:
                item["images"] = images
            converted.append(item)
        elif role == "tool":
            tool_name = (
                id_to_name.get(message.tool_call_id or "")
                or message.name
                or "unknown_tool"
            )
            if (
                max_tool_result_chars is not None
                and len(content) > max_tool_result_chars
            ):
                content = (
                    content[:max_tool_result_chars]
                    + TOOL_OUTPUT_TRUNCATED_SUFFIX
                )
            item = {"role": "tool", "content": content, "tool_name": tool_name}
            converted.append(item)
        else:
            item = {"role": role, "content": content}
            if images:
                item["images"] = images
            converted.append(item)
    return converted


def build_ollama_payload(
    request: ChatCompletionRequest,
    settings: Settings,
    tools: list[ToolSpec],
) -> dict[str, Any]:
    model = settings.resolve_model(request.model)
    messages = build_ollama_messages(
        request.messages,
        max_tool_result_chars=settings.max_tool_result_chars,
    )

    max_tokens = request.max_completion_tokens or request.max_tokens
    num_predict = min(
        max_tokens if max_tokens is not None else settings.max_output_tokens,
        settings.max_output_tokens,
    )
    options: dict[str, Any] = {
        "num_ctx": settings.num_ctx,
        "num_predict": num_predict,
    }
    temperature = (
        request.temperature if request.temperature is not None else settings.temperature
    )
    if temperature is not None:
        options["temperature"] = temperature
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if request.seed is not None:
        options["seed"] = request.seed
    if request.stop is not None:
        options["stop"] = request.stop

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if tools:
        payload["tools"] = [tool.to_ollama() for tool in tools]
    if settings.keep_alive:
        payload["keep_alive"] = settings.keep_alive
    return payload


def _arguments_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_images(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    images: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        image = part.get("image_url")
        if isinstance(image, dict):
            image = image.get("url")
        if isinstance(image, str) and image.startswith("data:") and ";base64," in image:
            images.append(image.split(";base64,", 1)[1])
    return images


@dataclass(frozen=True)
class ValidatedToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    @property
    def arguments_json(self) -> str:
        return json.dumps(
            self.arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments_json,
            },
        }


@dataclass
class Completion:
    id: str
    created: int
    model: str
    content: str | None
    tool_calls: list[ValidatedToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)

    def to_openai_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            message["tool_calls"] = [call.to_openai() for call in self.tool_calls]
        return message

    def to_openai_response(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": self.to_openai_message(),
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": self.usage,
        }

    def to_sse_chunks(self, *, include_usage: bool) -> list[dict[str, Any]]:
        base = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
        }
        chunks: list[dict[str, Any]] = []

        first_delta: dict[str, Any] = {"role": "assistant"}
        if self.content is not None:
            first_delta["content"] = self.content
        chunks.append(
            {
                **base,
                "choices": [
                    {"index": 0, "delta": first_delta, "finish_reason": None}
                ],
            }
        )

        if self.tool_calls:
            chunks.append(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": call.id,
                                        "type": "function",
                                        "function": {
                                            "name": call.name,
                                            "arguments": call.arguments_json,
                                        },
                                    }
                                    for index, call in enumerate(self.tool_calls)
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )

        chunks.append(
            {
                **base,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": self.finish_reason}
                ],
            }
        )
        if include_usage:
            chunks.append({**base, "choices": [], "usage": self.usage})
        return chunks


def new_completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def new_tool_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


def usage_from_ollama(response: dict[str, Any]) -> dict[str, int]:
    prompt = _as_int(response.get("prompt_eval_count"))
    completion = _as_int(response.get("eval_count"))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return 0


def finish_reason_from_ollama(response: dict[str, Any], *, has_tools: bool) -> str:
    if has_tools:
        return "tool_calls"
    reason = response.get("done_reason")
    if reason == "length":
        return "length"
    return "stop"


def tool_call_fingerprint(name: str, arguments: dict[str, Any]) -> str:
    canonical = json.dumps(
        arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{name}:{digest}"


def history_fingerprints(messages: list[ChatMessage]) -> list[tuple[int, set[str]]]:
    history: list[tuple[int, set[str]]] = []
    for index, message in enumerate(messages):
        if message.role != "assistant" or not message.tool_calls:
            continue
        fingerprints = {
            tool_call_fingerprint(
                call.function.name, _arguments_dict(call.function.arguments)
            )
            for call in message.tool_calls
        }
        history.append((index, fingerprints))
    return history


def count_tool_cycles(messages: list[ChatMessage]) -> int:
    return sum(
        1
        for message in messages
        if message.role == "assistant" and message.tool_calls
    )


def check_repeated_calls(
    calls: list[ParsedToolCall],
    messages: list[ChatMessage],
    *,
    repeat_limit: int,
) -> str | None:
    history = history_fingerprints(messages)
    if not history:
        return None

    last_index, last_fingerprints = history[-1]
    user_intervened = any(
        message.role == "user" for message in messages[last_index + 1 :]
    )

    counts: dict[str, int] = {}
    for _, fingerprints in history:
        for fingerprint in fingerprints:
            counts[fingerprint] = counts.get(fingerprint, 0) + 1

    for call in calls:
        fingerprint = tool_call_fingerprint(call.name, call.arguments)
        if not user_intervened and fingerprint in last_fingerprints:
            return SAFE_REPEAT
        if counts.get(fingerprint, 0) >= repeat_limit:
            return SAFE_REPEAT
    return None


def validate_calls(
    calls: list[ParsedToolCall],
    *,
    registry: ToolRegistry,
    guard: WorkspaceGuard,
    settings: Settings,
    messages: list[ChatMessage],
) -> tuple[list[ValidatedToolCall], str | None]:
    if not calls:
        return [], None

    # Resolve and normalize first so aliases such as file_path/path and
    # get_directory/read are deduplicated and compared consistently.
    normalized_calls: list[tuple[str, dict[str, Any], str]] = []
    seen: set[str] = set()
    for call in calls:
        resolved = registry.resolve_name(call.name)
        if resolved is None:
            return [], SAFE_UNKNOWN
        normalized = registry.normalize_arguments(
            resolved, call.arguments, original_name=call.name
        )
        fingerprint = tool_call_fingerprint(resolved, normalized)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        normalized_calls.append((resolved, normalized, call.source))

    if len(normalized_calls) > settings.max_tool_calls:
        return [], SAFE_LIMIT
    if count_tool_cycles(messages) >= settings.max_tool_cycles:
        return [], SAFE_CYCLE

    repeated = check_repeated_calls(
        [
            ParsedToolCall(name=name, arguments=arguments, source=source)
            for name, arguments, source in normalized_calls
        ],
        messages,
        repeat_limit=settings.repeat_call_limit,
    )
    if repeated:
        return [], repeated

    validated: list[ValidatedToolCall] = []
    for resolved, normalized, _source in normalized_calls:
        try:
            normalized = guard.validate_tool_arguments(resolved, normalized)
            registry.validate_schema(resolved, normalized)
        except SecurityError as exc:
            logger.warning(
                "tool call security rejection tool=%s reason=%s",
                resolved,
                exc.message,
            )
            return [], SAFE_SECURITY
        except ToolValidationError as exc:
            logger.warning(
                "tool call schema rejection tool=%s reason=%s",
                resolved,
                exc.message,
            )
            return [], SAFE_INVALID
        validated.append(
            ValidatedToolCall(
                id=new_tool_call_id(),
                name=resolved,
                arguments=normalized,
            )
        )
    return validated, None


def make_completion(
    *,
    model: str,
    content: str | None,
    tool_calls: list[ValidatedToolCall] | None = None,
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
) -> Completion:
    return Completion(
        id=new_completion_id(),
        created=int(time.time()),
        model=model,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        usage=usage
        or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
