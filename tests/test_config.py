from __future__ import annotations

import pytest

from adapter.config import ConfigError, Settings


def test_from_env_defaults(tmp_path) -> None:
    settings = Settings.from_env(
        {
            "ADAPTER_WORKSPACE": str(tmp_path),
            "ADAPTER_MODELS": "qwen2.5-coder:7b",
        }
    )
    assert settings.host == "127.0.0.1"
    assert settings.port == 8765
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.workspace == tmp_path.resolve()
    assert settings.allowed_models == ("qwen2.5-coder:7b",)
    assert settings.api_key is None
    assert settings.num_ctx == 16384
    assert settings.max_output_tokens == 4096
    assert settings.max_tool_result_chars == 20_000


def test_from_env_parses_optional_values(tmp_path) -> None:
    settings = Settings.from_env(
        {
            "ADAPTER_WORKSPACE": str(tmp_path),
            "ADAPTER_MODELS": "a, b, a",
            "ADAPTER_API_KEY": " token ",
            "ADAPTER_TEMPERATURE": "0.2",
            "ADAPTER_KEEP_ALIVE": "10m",
            "ADAPTER_LOG_LEVEL": "debug",
            "ADAPTER_MAX_TOOL_RESULT_CHARS": "1234",
        }
    )
    assert settings.allowed_models == ("a", "b")
    assert settings.api_key == "token"
    assert settings.temperature == pytest.approx(0.2)
    assert settings.keep_alive == "10m"
    assert settings.log_level == "DEBUG"
    assert settings.max_tool_result_chars == 1234


def test_public_host_is_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError, match="loopback"):
        Settings.from_env(
            {
                "ADAPTER_WORKSPACE": str(tmp_path),
                "ADAPTER_HOST": "0.0.0.0",
            }
        )


def test_missing_workspace_is_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        Settings.from_env(
            {"ADAPTER_WORKSPACE": str(tmp_path / "missing")}
        )


def test_invalid_port_is_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError, match="ADAPTER_PORT"):
        Settings.from_env(
            {"ADAPTER_WORKSPACE": str(tmp_path), "ADAPTER_PORT": "70000"}
        )


def test_invalid_model_list_is_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError, match="ADAPTER_MODELS"):
        Settings.from_env(
            {"ADAPTER_WORKSPACE": str(tmp_path), "ADAPTER_MODELS": " , "}
        )


def test_model_resolution(tmp_path) -> None:
    settings = Settings(
        workspace=tmp_path, allowed_models=("qwen2.5-coder:7b",)
    )
    assert settings.model_allowed("qwen2.5-coder:7b")
    assert settings.model_allowed("ollama-adapter/qwen2.5-coder:7b")
    assert not settings.model_allowed("other:latest")
    assert (
        settings.resolve_model("ollama-adapter/qwen2.5-coder:7b")
        == "qwen2.5-coder:7b"
    )
    with pytest.raises(ConfigError, match="not in ADAPTER_MODELS"):
        settings.resolve_model("other:latest")


def test_wildcard_model_resolution(tmp_path) -> None:
    settings = Settings(
        workspace=tmp_path,
        allowed_models=("*", "qwen2.5-coder:7b"),
    )
    assert settings.model_allowed("anything:latest")
    assert settings.resolve_model("anything:latest") == "anything:latest"
    assert (
        settings.resolve_model("ollama-adapter/anything:latest")
        == "anything:latest"
    )


def test_wildcard_only_is_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError, match="concrete model"):
        Settings(workspace=tmp_path, allowed_models=("*",))


def test_credentials_in_ollama_url_are_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError, match="credentials"):
        Settings(
            workspace=tmp_path,
            ollama_base_url="http://user:pass@127.0.0.1:11434",
        )
