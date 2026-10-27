from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .errors import SecurityError

_RESERVED_WINDOWS_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)

# These are deliberately conservative, defence-in-depth checks. They are not a
# sandbox: OpenCode still executes the command with the user's privileges. Use
# OpenCode permissions as the primary control.
_DESTRUCTIVE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\brm\s+(?:-[a-z]*[rf][a-z]*\s*)+"), "recursive/forced rm"),
    (re.compile(r"(?i)\b(?:remove-item|ri)\b[^\r\n;&|]*(?:-recurse|-r)\b"), "recursive Remove-Item"),
    (re.compile(r"(?i)\b(?:rd|rmdir|erase)\b[^\r\n;&|]*/s\b"), "recursive directory delete"),
    (re.compile(r"(?i)\bdel\b[^\r\n;&|]*/[fsq]\b"), "forced/wildcard delete"),
    (re.compile(r"(?i)(?:^|[;&|]\s*)\s*format(?:\.com)?\b"), "disk format"),
    (re.compile(r"(?i)\b(?:diskpart|clear-disk|initialize-disk|format-volume)\b"), "disk management"),
    (re.compile(r"(?i)\b(?:remove-partition|resize-partition|set-disk)\b"), "disk management"),
    (re.compile(r"(?i)\b(?:bcdedit|bootrec|vssadmin\s+delete|wbadmin\s+delete)\b"), "boot/recovery modification"),
    (re.compile(r"(?i)\b(?:shutdown|restart-computer|stop-computer|logoff)\b"), "system shutdown"),
    (re.compile(r"(?i)\bchkdsk\b[^\r\n;&|]*/f\b"), "filesystem repair"),
    (re.compile(r"(?i)\b(?:mkfs(?:\.\w+)?|dd\s+if=.*\bof=/dev/)\b"), "raw disk write"),
    (re.compile(r"(?i)\breg\s+delete\b"), "registry deletion"),
    (re.compile(r"(?i)\breg\s+add\b[^\r\n;&|]*\\run\b"), "registry autostart modification"),
    (re.compile(r"(?i)\b(?:sc|schtasks)\s+(?:/?(?:create|delete|change|config))\b"), "service/task modification"),
    (re.compile(r"(?i)\b(?:new-service|set-service|remove-service|register-scheduledtask|unregister-scheduledtask)\b"), "service/task modification"),
    (re.compile(r"(?i)\bnet\s+user\b[^\r\n;&|]*\b/add\b"), "local account creation"),
    (re.compile(r"(?i)\bnet\s+localgroup\s+administrators\b[^\r\n;&|]*\b/add\b"), "administrator group modification"),
    (re.compile(r"(?i)\b(?:set-executionpolicy|set-mppreference|add-mppreference|disable-windowsdefender)\b"), "security setting modification"),
    (re.compile(r"(?i)\bnetsh\s+(?:advfirewall|firewall)\b"), "firewall modification"),
    (re.compile(r"(?i)\b(?:takeown|icacls|cacls)\b[^\r\n;&|]*(?:/grant|/f\b|/r\b)"), "permission modification"),
    (re.compile(r"(?i)\bcipher\b[^\r\n;&|]*/w\b"), "free-space wiping"),
    (re.compile(r"(?i)\b(?:invoke-expression|iex)\b"), "dynamic code execution"),
    (re.compile(r"(?i)\bpowershell(?:\.exe)?\b[^\r\n;&|]*\s-(?:enc|encodedcommand)\b"), "encoded PowerShell"),
    (re.compile(r"(?i)\b(?:curl|wget|iwr|invoke-webrequest)\b[^\r\n;&|]*\|\s*(?:sh|bash|iex|powershell)\b"), "download-and-execute pipeline"),
    (re.compile(r"(?i)\b(?:certutil|bitsadmin)\b[^\r\n;&|]*(?:-urlcache|/transfer)\b"), "download utility abuse"),
    (re.compile(r"(?i)\b(?:wmic)\b[^\r\n;&|]*(?:process|delete|call\s+create)\b"), "WMI process/delete abuse"),
    (re.compile(r"(?i)\bgit\s+(?:clean\s+-[a-z]*[fd][a-z]*|reset\s+--hard|checkout\s+--\s+\.|restore\s+\.)\b"), "destructive git operation"),
    (re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:"), "fork bomb"),
    (re.compile(r"(?i)\b(?:attrib)\s+\+[sh]\b"), "hidden/system attribute change"),
)

_CD_RE = re.compile(
    r"(?i)(?:^|[;&|]\s*)\s*(?:cd|chdir|set-location|sl|pushd)\s+"
    r"(?:/[dD]\s+)?"
    r"(?:(?:-path|-literalpath)\s+)?"
    r"(?P<path>\"[^\"]+\"|'[^']+'|[^\s;&|]+)"
)
_BARE_CD_RE = re.compile(
    r"(?i)(?:^|[;&|]\s*)\s*(?:cd|chdir|set-location|sl|pushd)\s*(?:[;&|]|$)"
)
_PATH_ARG_RE = re.compile(
    r"(?i)(?:-path|-filepath|-literalpath|-destination|-outfile|-target)\s+"
    r"(?P<path>\"[^\"]+\"|'[^']+'|[^\s;&|]+)"
)
_REDIRECT_RE = re.compile(
    r"(?<![>])>{1,2}\s*(?P<path>\"[^\"]+\"|'[^']+'|[^\s;&|]+)"
)


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


class WorkspaceGuard:
    """Validates model-generated file paths and shell-call arguments.

    The guard is intentionally strict: ``..`` components are rejected even when
    they would resolve back inside the workspace, because the adapter cannot
    know how OpenCode will interpret them.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        max_command_timeout_ms: int = 120_000,
        max_command_length: int = 8000,
    ) -> None:
        self.root = Path(workspace).resolve(strict=True)
        self.max_command_timeout_ms = max_command_timeout_ms
        self.max_command_length = max_command_length

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def validate_path(self, raw: Any, *, field: str = "path") -> Path:
        if not isinstance(raw, str):
            raise SecurityError(f"{field} must be a string")
        if "\x00" in raw:
            raise SecurityError(f"{field} contains a NUL byte")
        value = raw.strip()
        if not value:
            raise SecurityError(f"{field} must not be empty")
        if len(value) > 4096:
            raise SecurityError(f"{field} is too long")
        if "://" in value:
            raise SecurityError(f"{field} must be a filesystem path, not a URL")
        if value.startswith(("\\\\", "//")):
            raise SecurityError(f"{field} must not be a UNC/device path")
        if value.startswith("~"):
            raise SecurityError(f"{field} must not use home-directory expansion")
        if re.match(r"^[A-Za-z]:(?![\\/])", value):
            raise SecurityError(f"{field} must not be a drive-relative path")

        rest = value[2:] if re.match(r"^[A-Za-z]:", value) else value
        if ":" in rest:
            raise SecurityError(f"{field} must not contain a drive or stream separator")
        if "*" in value or "?" in value:
            raise SecurityError(f"{field} must not contain wildcard characters")

        parts = [part for part in re.split(r"[\\/]+", value) if part not in {"", "."}]
        if any(part == ".." for part in parts):
            raise SecurityError(f"{field} must not contain '..' path traversal")
        if any(part != part.rstrip(" .") for part in parts):
            raise SecurityError(f"{field} must not end path components with spaces or dots")
        for part in parts:
            base = part.split(".", 1)[0].upper()
            if base in _RESERVED_WINDOWS_NAMES:
                raise SecurityError(f"{field} uses a reserved Windows device name")

        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise SecurityError(f"{field} could not be resolved") from exc

        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise SecurityError(f"{field} escapes the configured workspace") from exc

        normalized_root = os.path.normcase(str(self.root))
        normalized_resolved = os.path.normcase(str(resolved))
        if normalized_resolved != normalized_root and not normalized_resolved.startswith(
            normalized_root + os.sep
        ):
            raise SecurityError(f"{field} escapes the configured workspace")
        return resolved

    def validate_glob_pattern(self, raw: Any, *, field: str = "pattern") -> None:
        if not isinstance(raw, str):
            raise SecurityError(f"{field} must be a string")
        if "\x00" in raw:
            raise SecurityError(f"{field} contains a NUL byte")
        value = raw.strip()
        if not value:
            raise SecurityError(f"{field} must not be empty")
        if len(value) > 2000:
            raise SecurityError(f"{field} is too long")
        if value.startswith(("\\\\", "//", "/", "\\", "~")):
            raise SecurityError(f"{field} must be a relative glob pattern")
        if re.match(r"^[A-Za-z]:", value):
            raise SecurityError(f"{field} must be a relative glob pattern")
        if ":" in value:
            raise SecurityError(f"{field} must not contain a drive or stream separator")
        parts = [part for part in re.split(r"[\\/]+", value) if part]
        if any(part == ".." for part in parts):
            raise SecurityError(f"{field} must not contain '..' path traversal")

    # ------------------------------------------------------------------
    # Shell calls
    # ------------------------------------------------------------------
    def validate_command(self, command: Any) -> None:
        if not isinstance(command, str):
            raise SecurityError("command must be a string")
        if "\x00" in command:
            raise SecurityError("command contains a NUL byte")
        if not command.strip():
            raise SecurityError("command must not be empty")
        if len(command) > self.max_command_length:
            raise SecurityError("command is longer than the configured limit")

        for pattern, reason in _DESTRUCTIVE_PATTERNS:
            if pattern.search(command):
                raise SecurityError(f"command blocked by security policy ({reason})")

        self._validate_command_paths(command)

    def _validate_command_paths(self, command: str) -> None:
        if _BARE_CD_RE.search(command):
            raise SecurityError(
                "bare directory changes are blocked; use workdir inside the workspace"
            )

        for match in _CD_RE.finditer(command):
            target = _unquote(match.group("path"))
            self._validate_command_path(target, "directory change")

        for match in _PATH_ARG_RE.finditer(command):
            target = _unquote(match.group("path"))
            if target.startswith("-"):
                continue
            self._validate_command_path(target, "command path argument")

        for match in _REDIRECT_RE.finditer(command):
            target = _unquote(match.group("path"))
            self._validate_command_path(target, "redirection target")

    def _validate_command_path(self, target: str, context: str) -> None:
        if not target:
            return
        if target.startswith(("$", "%", "~")):
            raise SecurityError(
                f"{context} cannot be verified because it uses variable or home expansion"
            )
        # Command flags such as /d or -Force are not paths. A Unix-style
        # absolute path is not a flag and must still be checked.
        if target.startswith("-"):
            return
        if target.startswith("/") and len(target) <= 3 and target[1:].isalpha():
            return
        self.validate_path(target, field=context)

    def validate_tool_arguments(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate a normalized tool call and return possibly clamped arguments."""
        args = dict(arguments)

        if tool_name in {"read", "write", "edit", "glob", "grep"}:
            path = args.get("path")
            if path is not None:
                self.validate_path(path, field="path")

        if tool_name == "shell":
            command = args.get("command")
            if command is None:
                raise SecurityError("shell call is missing a command")
            self.validate_command(command)
            workdir = args.get("workdir")
            if workdir is not None:
                self.validate_path(workdir, field="workdir")
            if args.get("background") is True:
                raise SecurityError(
                    "background shell commands are disabled by the adapter security policy"
                )
            timeout = args.get("timeout")
            if timeout is not None:
                if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                    raise SecurityError("shell timeout must be a number")
                if timeout <= 0:
                    args["timeout"] = self.max_command_timeout_ms
                elif timeout > self.max_command_timeout_ms:
                    args["timeout"] = self.max_command_timeout_ms

        if tool_name == "glob":
            pattern = args.get("pattern")
            if pattern is not None:
                self.validate_glob_pattern(pattern, field="pattern")

        if tool_name == "grep":
            include = args.get("include")
            if include is not None:
                self.validate_glob_pattern(include, field="include")

        if tool_name == "webfetch":
            url = args.get("url")
            if url is not None:
                self.validate_url(url)

        self._check_length(args, "websearch", "query", 2000)
        self._check_length(args, "subagent", "prompt", 20000)
        self._check_length(args, "question", "question", 4000)
        self._check_length(args, "skill", "id", 200)
        self._check_length(args, "execute", "code", 100000)

        return args

    @staticmethod
    def validate_url(raw: Any) -> None:
        if not isinstance(raw, str):
            raise SecurityError("url must be a string")
        value = raw.strip()
        if not value or len(value) > 2048:
            raise SecurityError("url is empty or too long")
        parsed = urlparse(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise SecurityError("url must use http:// or https://")
        if parsed.username or parsed.password:
            raise SecurityError("url must not contain credentials")

    @staticmethod
    def _check_length(args: dict[str, Any], tool_name: str, key: str, limit: int) -> None:
        value = args.get(key)
        if isinstance(value, str) and len(value) > limit:
            raise SecurityError(f"{tool_name}.{key} is longer than the configured limit")
