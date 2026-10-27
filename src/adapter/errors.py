from __future__ import annotations

from typing import Any


class AdapterError(Exception):
    """An error that can be rendered as an OpenAI-compatible error payload."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.param = param

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


class UpstreamError(AdapterError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        code: str = "upstream_error",
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            error_type="upstream_error",
            code=code,
        )


class SecurityError(AdapterError):
    def __init__(self, message: str, *, code: str = "security_policy") -> None:
        super().__init__(
            message,
            status_code=400,
            error_type="invalid_request_error",
            code=code,
        )


class ToolValidationError(AdapterError):
    def __init__(self, message: str, *, code: str = "invalid_tool_call") -> None:
        super().__init__(
            message,
            status_code=400,
            error_type="invalid_request_error",
            code=code,
        )
