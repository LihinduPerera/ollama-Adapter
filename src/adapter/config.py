from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

DEFAULT_MODEL = "qwen2.5-coder:7b"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class ConfigError(ValueError):
    """Raised when adapter configuration is missing or unsafe."""


def _env(env: Mapping[str, str], name: str, default: str | None = None) -> str | None:
    raw = env.get(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def _int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = _env(env, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return value


def _float(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = _env(env, name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return value


def _optional_float(
    env: Mapping[str, str],
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    raw = _env(env, name)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return value


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _env(env, name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false")


@dataclass(frozen=True)
class Settings:
    """Validated adapter settings.

    Construct directly in tests, or use :meth:`from_env` for environment-based
    configuration.
    """

    host: str = "127.0.0.1"
    port: int = 8765
    ollama_base_url: str = "http://127.0.0.1:11434"
    workspace: Path = field(default_factory=Path.cwd)
    api_key: str | None = None
    allowed_models: tuple[str, ...] = (DEFAULT_MODEL,)
    num_ctx: int = 16384
    max_output_tokens: int = 4096
    request_timeout: float = 300.0
    max_tool_calls: int = 4
    max_tool_cycles: int = 12
    repeat_call_limit: int = 2
    max_request_bytes: int = 2_000_000
    max_response_bytes: int = 8_000_000
    max_tool_result_chars: int = 20_000
    max_command_timeout_ms: int = 120_000
    max_command_length: int = 8000
    log_level: str = "INFO"
    keep_alive: str | None = None
    temperature: float | None = None
    health_timeout: float = 2.0

    def __post_init__(self) -> None:
        if self.host not in _LOOPBACK_HOSTS:
            raise ConfigError(
                "ADAPTER_HOST must be a loopback address "
                "(127.0.0.1, ::1, or localhost); refusing to bind publicly"
            )
        if not 1 <= self.port <= 65535:
            raise ConfigError("ADAPTER_PORT must be between 1 and 65535")

        workspace = Path(self.workspace)
        try:
            workspace = workspace.resolve(strict=True)
        except OSError as exc:
            raise ConfigError(f"ADAPTER_WORKSPACE does not exist: {workspace}") from exc
        if not workspace.is_dir():
            raise ConfigError(f"ADAPTER_WORKSPACE is not a directory: {workspace}")
        object.__setattr__(self, "workspace", workspace)

        base_url = self.ollama_base_url.rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigError("OLLAMA_BASE_URL must be an http:// or https:// URL")
        if parsed.username or parsed.password:
            raise ConfigError("OLLAMA_BASE_URL must not contain credentials")
        object.__setattr__(self, "ollama_base_url", base_url)

        models = tuple(
            dict.fromkeys(str(model).strip() for model in self.allowed_models if str(model).strip())
        )
        if not models:
            raise ConfigError("ADAPTER_MODELS must contain at least one model name")
        if models == ("*",):
            raise ConfigError(
                "ADAPTER_MODELS must include at least one concrete model when using *"
            )
        object.__setattr__(self, "allowed_models", models)

        if self.api_key is not None:
            key = self.api_key.strip()
            object.__setattr__(self, "api_key", key or None)

        if self.keep_alive is not None:
            keep_alive = self.keep_alive.strip()
            object.__setattr__(self, "keep_alive", keep_alive or None)

        if not 2048 <= self.num_ctx <= 131072:
            raise ConfigError("ADAPTER_NUM_CTX must be between 2048 and 131072")
        if not 1 <= self.max_output_tokens <= 32768:
            raise ConfigError("ADAPTER_MAX_OUTPUT_TOKENS must be between 1 and 32768")
        if not 10.0 <= self.request_timeout <= 1800.0:
            raise ConfigError("ADAPTER_REQUEST_TIMEOUT must be between 10 and 1800 seconds")
        if not 1 <= self.max_tool_calls <= 16:
            raise ConfigError("ADAPTER_MAX_TOOL_CALLS must be between 1 and 16")
        if not 1 <= self.max_tool_cycles <= 100:
            raise ConfigError("ADAPTER_MAX_TOOL_CYCLES must be between 1 and 100")
        if not 1 <= self.repeat_call_limit <= 10:
            raise ConfigError("ADAPTER_REPEAT_CALL_LIMIT must be between 1 and 10")
        if not 1024 <= self.max_request_bytes <= 50_000_000:
            raise ConfigError("ADAPTER_MAX_REQUEST_BYTES must be between 1024 and 50000000")
        if not 1024 <= self.max_response_bytes <= 50_000_000:
            raise ConfigError("ADAPTER_MAX_RESPONSE_BYTES must be between 1024 and 50000000")
        if not 1000 <= self.max_tool_result_chars <= 500_000:
            raise ConfigError(
                "ADAPTER_MAX_TOOL_RESULT_CHARS must be between 1000 and 500000"
            )
        if not 1000 <= self.max_command_timeout_ms <= 600_000:
            raise ConfigError(
                "ADAPTER_MAX_COMMAND_TIMEOUT_MS must be between 1000 and 600000"
            )
        if not 64 <= self.max_command_length <= 100_000:
            raise ConfigError("ADAPTER_MAX_COMMAND_LENGTH must be between 64 and 100000")
        if self.log_level.upper() not in _LOG_LEVELS:
            raise ConfigError("ADAPTER_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        object.__setattr__(self, "log_level", self.log_level.upper())
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise ConfigError("ADAPTER_TEMPERATURE must be between 0 and 2")
        if not 0.5 <= self.health_timeout <= 30.0:
            raise ConfigError("ADAPTER_HEALTH_TIMEOUT must be between 0.5 and 30 seconds")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if env is None else env
        workspace = Path(_env(values, "ADAPTER_WORKSPACE", str(Path.cwd())) or Path.cwd())
        models_raw = _env(values, "ADAPTER_MODELS", DEFAULT_MODEL) or DEFAULT_MODEL
        models = tuple(
            dict.fromkeys(part.strip() for part in models_raw.split(",") if part.strip())
        )
        keep_alive = _env(values, "ADAPTER_KEEP_ALIVE")
        return cls(
            host=_env(values, "ADAPTER_HOST", "127.0.0.1") or "127.0.0.1",
            port=_int(values, "ADAPTER_PORT", 8765, minimum=1, maximum=65535),
            ollama_base_url=_env(
                values, "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            )
            or "http://127.0.0.1:11434",
            workspace=workspace,
            api_key=_env(values, "ADAPTER_API_KEY"),
            allowed_models=models,
            num_ctx=_int(values, "ADAPTER_NUM_CTX", 16384, minimum=2048, maximum=131072),
            max_output_tokens=_int(
                values, "ADAPTER_MAX_OUTPUT_TOKENS", 4096, minimum=1, maximum=32768
            ),
            request_timeout=_float(
                values, "ADAPTER_REQUEST_TIMEOUT", 300.0, minimum=10.0, maximum=1800.0
            ),
            max_tool_calls=_int(
                values, "ADAPTER_MAX_TOOL_CALLS", 4, minimum=1, maximum=16
            ),
            max_tool_cycles=_int(
                values, "ADAPTER_MAX_TOOL_CYCLES", 12, minimum=1, maximum=100
            ),
            repeat_call_limit=_int(
                values, "ADAPTER_REPEAT_CALL_LIMIT", 2, minimum=1, maximum=10
            ),
            max_request_bytes=_int(
                values,
                "ADAPTER_MAX_REQUEST_BYTES",
                2_000_000,
                minimum=1024,
                maximum=50_000_000,
            ),
            max_response_bytes=_int(
                values,
                "ADAPTER_MAX_RESPONSE_BYTES",
                8_000_000,
                minimum=1024,
                maximum=50_000_000,
            ),
            max_tool_result_chars=_int(
                values,
                "ADAPTER_MAX_TOOL_RESULT_CHARS",
                20_000,
                minimum=1000,
                maximum=500_000,
            ),
            max_command_timeout_ms=_int(
                values,
                "ADAPTER_MAX_COMMAND_TIMEOUT_MS",
                120_000,
                minimum=1000,
                maximum=600_000,
            ),
            max_command_length=_int(
                values,
                "ADAPTER_MAX_COMMAND_LENGTH",
                8000,
                minimum=64,
                maximum=100_000,
            ),
            log_level=_env(values, "ADAPTER_LOG_LEVEL", "INFO") or "INFO",
            keep_alive=keep_alive,
            temperature=_optional_float(
                values, "ADAPTER_TEMPERATURE", minimum=0.0, maximum=2.0
            ),
            health_timeout=_float(
                values, "ADAPTER_HEALTH_TIMEOUT", 2.0, minimum=0.5, maximum=30.0
            ),
        )

    def model_allowed(self, model: str) -> bool:
        if "*" in self.allowed_models:
            return True
        if model in self.allowed_models:
            return True
        if "/" in model:
            return model.split("/", 1)[1] in self.allowed_models
        return False

    def resolve_model(self, model: str) -> str:
        """Return the Ollama model name, accepting an optional provider prefix."""
        if model in self.allowed_models:
            return model
        if "/" in model:
            suffix = model.split("/", 1)[1]
            if suffix in self.allowed_models:
                return suffix
            if "*" in self.allowed_models:
                return suffix
        if "*" in self.allowed_models:
            return model
        raise ConfigError(
            "Model is not in ADAPTER_MODELS. "
            f"Allowed: {', '.join(self.allowed_models)}"
        )
