from __future__ import annotations

import json
import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .config import ConfigError, Settings
from .errors import AdapterError
from .logging_config import configure_logging
from .ollama_client import OllamaClient
from .protocol import (
    SAFE_INVALID,
    SAFE_NO_TOOLS,
    ChatCompletionRequest,
    Completion,
    ToolRegistry,
    build_ollama_payload,
    filter_tools,
    finish_reason_from_ollama,
    make_completion,
    usage_from_ollama,
    validate_calls,
)
from .security import WorkspaceGuard
from .tool_parser import content_to_text, parse_tool_calls

logger = logging.getLogger(__name__)


class ChatService:
    """Translates one OpenAI chat request into Ollama and back."""

    def __init__(self, settings: Settings, client: OllamaClient) -> None:
        self.settings = settings
        self.client = client
        self.guard = WorkspaceGuard(
            settings.workspace,
            max_command_timeout_ms=settings.max_command_timeout_ms,
            max_command_length=settings.max_command_length,
        )

    async def complete(self, request: ChatCompletionRequest) -> Completion:
        try:
            self.settings.resolve_model(request.model)
        except ConfigError as exc:
            raise AdapterError(
                str(exc),
                status_code=404,
                error_type="invalid_request_error",
                code="model_not_allowed",
            ) from exc

        tools = filter_tools(request.tools, request.tool_choice)
        registry = ToolRegistry(tools)
        payload = build_ollama_payload(
            request, self.settings, list(registry.specs.values())
        )
        upstream = await self.client.chat(payload)
        message = upstream["message"]
        usage = usage_from_ollama(upstream)
        raw_content = content_to_text(message.get("content"))

        if not registry.advertised:
            probe = parse_tool_calls(message, tools_advertised=True)
            if probe.had_tool_syntax:
                return make_completion(
                    model=request.model,
                    content=SAFE_NO_TOOLS,
                    usage=usage,
                )
            return make_completion(
                model=request.model,
                content=raw_content,
                finish_reason=finish_reason_from_ollama(upstream, has_tools=False),
                usage=usage,
            )

        parsed = parse_tool_calls(message, tools_advertised=True)
        if parsed.errors:
            logger.warning(
                "tool call rejected: parser errors=%d structured=%s",
                len(parsed.errors),
                parsed.structured,
            )
            return make_completion(
                model=request.model,
                content=SAFE_INVALID,
                usage=usage,
            )

        if parsed.calls:
            validated, error = validate_calls(
                parsed.calls,
                registry=registry,
                guard=self.guard,
                settings=self.settings,
                messages=request.messages,
            )
            if error:
                logger.warning("tool call rejected: %s", error)
                return make_completion(
                    model=request.model,
                    content=error,
                    usage=usage,
                )
            return make_completion(
                model=request.model,
                content=parsed.text.strip() or None,
                tool_calls=validated,
                finish_reason="tool_calls",
                usage=usage,
            )

        if parsed.had_tool_syntax:
            logger.warning("tool call rejected: tool syntax produced no valid calls")
            return make_completion(
                model=request.model,
                content=SAFE_INVALID,
                usage=usage,
            )

        return make_completion(
            model=request.model,
            content=parsed.text,
            finish_reason=finish_reason_from_ollama(upstream, has_tools=False),
            usage=usage,
        )


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """ASGI middleware that rejects oversized request bodies.

    Content-Length is checked first. Chunked bodies are counted as they arrive
    so a client cannot bypass the limit by omitting the header.
    """

    def __init__(self, app: Any, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            key.lower(): value for key, value in (scope.get("headers") or [])
        }
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                length = int(content_length)
            except ValueError:
                length = 0
            if length > self.max_bytes:
                await self._reject(send, 413)
                return

        received = 0
        started = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal started
            if message.get("type") == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _BodyTooLarge:
            if not started:
                await self._reject(send, 413)

    @staticmethod
    async def _reject(send: Any, status_code: int) -> None:
        body = json.dumps(
            {
                "error": {
                    "message": "Request body is too large",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "request_too_large",
                }
            }
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    if app.state.own_client:
        await app.state.ollama.close()


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or uuid.uuid4().hex


async def _sse_stream(
    completion: Completion, *, include_usage: bool
) -> AsyncIterator[str]:
    for chunk in completion.to_sse_chunks(include_usage=include_usage):
        yield "data: " + json.dumps(
            chunk, ensure_ascii=False, separators=(",", ":")
        ) + "\n\n"
    yield "data: [DONE]\n\n"


def create_app(
    settings: Settings | None = None,
    ollama_client: OllamaClient | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    own_client = ollama_client is None
    client = ollama_client or OllamaClient(
        settings.ollama_base_url,
        timeout=settings.request_timeout,
        max_response_bytes=settings.max_response_bytes,
    )

    app = FastAPI(
        title="OpenCode Ollama Tool Adapter",
        version=__version__,
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.ollama = client
    app.state.own_client = own_client
    app.state.service = ChatService(settings, client)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)

    async def require_auth(request: Request) -> None:
        api_key = settings.api_key
        if not api_key:
            return
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            token.strip(), api_key
        ):
            raise AdapterError(
                "Invalid or missing API key",
                status_code=401,
                error_type="authentication_error",
                code="invalid_api_key",
            )

    @app.exception_handler(AdapterError)
    async def adapter_error_handler(
        request: Request, exc: AdapterError
    ) -> JSONResponse:
        request_id = _request_id(request)
        logger.warning(
            "request failed request_id=%s status=%d code=%s",
            request_id,
            exc.status_code,
            exc.code,
        )
        headers = {"x-request-id": request_id}
        if exc.status_code == 401:
            headers["WWW-Authenticate"] = "Bearer"
        return JSONResponse(
            exc.to_payload(), status_code=exc.status_code, headers=headers
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details: list[str] = []
        for error in exc.errors()[:5]:
            location = ".".join(str(part) for part in error.get("loc", []))
            message = str(error.get("msg", "invalid value"))
            details.append(f"{location}: {message}" if location else message)
        request_id = _request_id(request)
        return JSONResponse(
            {
                "error": {
                    "message": "Invalid request: " + "; ".join(details),
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_request",
                }
            },
            status_code=400,
            headers={"x-request-id": request_id},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        request_id = _request_id(request)
        logger.exception("unhandled adapter error request_id=%s", request_id)
        return JSONResponse(
            {
                "error": {
                    "message": "Internal adapter error",
                    "type": "server_error",
                    "param": None,
                    "code": "internal_error",
                }
            },
            status_code=500,
            headers={"x-request-id": request_id},
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        ollama = await client.health(timeout=settings.health_timeout)
        return {
            "status": "ok" if ollama.get("reachable") else "degraded",
            "adapter": "ok",
            "version": __version__,
            "ollama": ollama,
            "models": list(settings.allowed_models),
        }

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "ollama-adapter",
                }
                for model in settings.allowed_models
                if model != "*"
            ],
        }

    @app.post(
        "/v1/chat/completions",
        dependencies=[Depends(require_auth)],
        response_model=None,
    )
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        request_id = _request_id(request)
        started = time.perf_counter()
        logger.info(
            "chat start request_id=%s model=%s messages=%d tools=%d stream=%s",
            request_id,
            body.model,
            len(body.messages),
            len(body.tools or []),
            body.stream,
        )
        completion = await app.state.service.complete(body)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "chat done request_id=%s model=%s finish=%s tool_calls=%d elapsed_ms=%d",
            request_id,
            body.model,
            completion.finish_reason,
            len(completion.tool_calls),
            elapsed_ms,
        )

        if body.stream:
            include_usage = bool(
                body.stream_options and body.stream_options.include_usage
            )
            return StreamingResponse(
                _sse_stream(completion, include_usage=include_usage),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "x-request-id": request_id,
                },
            )

        return JSONResponse(
            completion.to_openai_response(),
            headers={"x-request-id": request_id},
        )

    return app


def run() -> None:
    load_dotenv()
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    configure_logging(settings.log_level, settings.api_key)
    logger.info(
        "starting adapter host=%s port=%d workspace=%s models=%s",
        settings.host,
        settings.port,
        settings.workspace,
        ",".join(settings.allowed_models),
    )
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        log_config=None,
        access_log=True,
    )


if __name__ == "__main__":
    run()
