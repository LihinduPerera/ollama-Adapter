# OpenCode Ollama Tool Adapter

A localhost-only Python adapter that lets **OpenCode 2.1.4** use local **Ollama 0.40.11** models whose tool calls arrive as text instead of structured `tool_calls`.

The adapter exposes an OpenAI-compatible `POST /v1/chat/completions` endpoint, forwards the conversation to Ollama's native `POST /api/chat`, parses tool calls from the model output, validates them, and returns real OpenAI `tool_calls` (JSON or SSE). **OpenCode still executes every tool.** The adapter does not reimplement OpenCode's tools.

Verified end to end with `qwen2.5-coder:7b`: OpenCode read a file, wrote a file, edited a file, and ran `python -V` through the adapter.

---

## Why this exists

OpenCode normally expects an OpenAI-compatible model to return tool calls in `message.tool_calls` (or in streaming `delta.tool_calls`). Qwen2.5-Coder 7B in Ollama can instead return a bare JSON object in `message.content`:

```json
{"name": "get_directory", "arguments": {"path": "."}}
```

Ollama reports no `tool_calls`, so OpenCode sees ordinary text and does not execute anything. The adapter bridges that gap:

```text
OpenCode
   |
   |  OpenAI chat completions (tools advertised)
   v
Adapter 127.0.0.1:8765
   |
   |  Ollama native /api/chat, stream=false
   v
Ollama 127.0.0.1:11434
   |
   |  content with bare JSON / <tool_call> / structured calls
   v
Adapter: parse -> alias -> validate -> OpenAI tool_calls
   |
   v
OpenCode executes the real tool and sends the result back
```

---

## Requirements

| Component | Tested version |
|---|---|
| Windows | Windows 11, PowerShell 5.1 |
| Python | 3.10.0 |
| OpenCode | 2.1.4 |
| Ollama | 0.40.11 |
| Model | `qwen2.5-coder:7b` |
| VRAM | 8 GB (context 16384, output 4096) |

No Ollama upgrade, OpenCode upgrade, or additional large model is required.

---

## Quick start

Run these commands from the repository root in PowerShell.

### 1. Install the adapter

```powershell
.\scripts\setup.ps1
```

This creates `.venv`, installs the pinned dependencies, creates `.env` from `.env.example` if needed, and runs the test suite.

If PowerShell blocks local scripts for this session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 2. Make sure Ollama is running

```powershell
ollama list
Invoke-RestMethod http://127.0.0.1:11434/api/version
```

### 3. Start the adapter for one project

```powershell
.\scripts\start-adapter.ps1 -Workspace "D:\path\to\your\project"
```

The adapter binds only to `127.0.0.1:8765`. It refuses to start on `0.0.0.0`.

Check it:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
Invoke-RestMethod http://127.0.0.1:8765/v1/models
```

### 4. Add the provider to OpenCode

The installer backs up the existing config before changing it:

```powershell
.\scripts\install-opencode-provider.ps1
```

It adds an `ollama-adapter` provider to:

```text
C:\Users\<you>\.config\opencode\opencode.json
```

The existing `providers.ollama` section and the existing default model are preserved. A backup is written next to the config as `opencode.json.bak-<timestamp>`. Restart OpenCode if it was already running.

Alternatively, copy [`examples/opencode.json`](examples/opencode.json) into a project as `opencode.json`, or merge its `provider` block into your global config.

### 5. Verify OpenCode sees the model

```powershell
opencode models
```

You should see:

```text
ollama-adapter/qwen2.5-coder:7b
```

### 6. Use it

Non-interactive:

```powershell
opencode run --auto -m ollama-adapter/qwen2.5-coder:7b "Read README.md and summarize it"
```

Interactive: start `opencode`, open the model picker, and select **Ollama Tool Adapter / Qwen2.5 Coder 7B**.

`--auto` auto-approves tool permissions. Without it, OpenCode asks before executing tools; that is the safer default.

The global default model is intentionally left as `ollama/qwen2.5-coder:7b` so the direct Ollama setup keeps working when the adapter is not running. Select `ollama-adapter/qwen2.5-coder:7b` when you want tool execution. To make it the default, change the top-level `model` value to `ollama-adapter/qwen2.5-coder:7b`.

---

## Configuration

Settings come from environment variables or a `.env` file in the repository root. The start script sets the important ones per process.

| Variable | Default | Purpose |
|---|---|---|
| `ADAPTER_HOST` | `127.0.0.1` | Loopback bind address only. |
| `ADAPTER_PORT` | `8765` | Local HTTP port. |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama native API. |
| `ADAPTER_WORKSPACE` | current directory | Root that model-generated file paths must stay inside. |
| `ADAPTER_MODELS` | `qwen2.5-coder:7b` | Comma-separated allow-list. |
| `ADAPTER_API_KEY` | empty | Optional bearer token for `/v1/*`. |
| `ADAPTER_NUM_CTX` | `16384` | Ollama context window. |
| `ADAPTER_MAX_OUTPUT_TOKENS` | `4096` | Upper bound on generated tokens. |
| `ADAPTER_REQUEST_TIMEOUT` | `300` | Ollama request timeout, seconds. |
| `ADAPTER_MAX_TOOL_CALLS` | `4` | Maximum calls in one model response. |
| `ADAPTER_MAX_TOOL_CYCLES` | `12` | Maximum assistant tool-call cycles in one conversation. |
| `ADAPTER_REPEAT_CALL_LIMIT` | `2` | Repeated-call guard. |
| `ADAPTER_MAX_REQUEST_BYTES` | `2000000` | Request body limit. |
| `ADAPTER_MAX_RESPONSE_BYTES` | `8000000` | Ollama response limit. |
| `ADAPTER_MAX_TOOL_RESULT_CHARS` | `20000` | Tool-result characters forwarded to Ollama; larger results are truncated. |
| `ADAPTER_MAX_COMMAND_TIMEOUT_MS` | `120000` | Clamp for shell `timeout`. |
| `ADAPTER_MAX_COMMAND_LENGTH` | `8000` | Shell command length limit. |
| `ADAPTER_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `ADAPTER_KEEP_ALIVE` | empty | Optional Ollama `keep_alive`, e.g. `5m`. |
| `ADAPTER_TEMPERATURE` | empty | Optional generation temperature. |

`ADAPTER_MODELS=*` is accepted only together with at least one concrete model, so `/v1/models` has a selectable model to advertise.

**One workspace per adapter process.** The workspace guard is fixed at startup. If you switch to a different project, restart the adapter with that project's path:

```powershell
.\scripts\start-adapter.ps1 -Workspace "D:\projects\other-project"
```

### Optional API key

Generate a token:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Set `ADAPTER_API_KEY` in `.env` (or pass `-ApiKey` to the start script). Then OpenCode must send the same token. With the `@ai-sdk/openai-compatible` provider, add `apiKey` under `options`:

```json
"options": {
  "baseURL": "http://127.0.0.1:8765/v1",
  "apiKey": "your-token"
}
```

`/health` stays public so local health checks work.

---

## What the adapter translates

### Input

It accepts the OpenAI chat-completions shape OpenCode sends, including:

- `messages` with `system`, `user`, `assistant`, and `tool` roles
- assistant `tool_calls` and tool results
- `tools`, `tool_choice`, `stream`, `stream_options.include_usage`
- `max_completion_tokens` / `max_tokens`, `temperature`, `top_p`, `stop`, `seed`

The request is forwarded to Ollama with `stream: false`. The adapter buffers the full response, then emits either a normal OpenAI JSON response or a valid OpenAI SSE stream.

### Tool-call forms parsed

1. Ollama structured `message.tool_calls`
2. `<tool_call>...</tool_call>` blocks, including multiple JSON values
3. Bare JSON that is the entire assistant content and is call-shaped
4. JSON arrays of calls
5. Fenced bare JSON

A bare JSON object is treated as a tool call only when the whole trimmed content is JSON and it contains a call-shaped key (`arguments`, `parameters`, `function`, `tool_calls`, `tool`, or `tool_name`). Ordinary JSON in an explanation is not executed.

### Name aliases

Common model-generated names are mapped to OpenCode's tool names, but only when the target tool is actually advertised in the current request:

| Generated name examples | OpenCode tool |
|---|---|
| `get_directory`, `list_directory`, `ls`, `read_file`, `cat` | `read` |
| `write_file`, `create_file`, `save_file` | `write` |
| `edit_file`, `replace_in_file`, `str_replace` | `edit` |
| `run_command`, `bash`, `powershell`, `terminal` | `shell` |
| `glob_files`, `find_files` | `glob` |
| `search_files`, `ripgrep` | `grep` |

### Argument aliases

For advertised tools, common argument names are normalized:

- `file_path` / `filename` / `file` -> `path`
- `text` / `contents` -> `content`
- `old_string` -> `oldString`
- `new_string` -> `newString`
- `replace_all` -> `replaceAll`
- `cmd` -> `command`
- `cwd` / `directory` -> `workdir`

The normalized arguments are validated against the **actual JSON Schema OpenCode advertised** for that tool. Unknown tools, duplicate calls, calls over the limit, and schema violations are never returned for execution.

### Output

- Non-streaming: OpenAI `chat.completion` JSON
- Streaming: `role` delta, content delta, `tool_calls` delta with `index`/`id`/`type`/`function`, finish reason, optional usage chunk, `data: [DONE]`
- If no tools were advertised, tool syntax is suppressed instead of leaking a call
- If parsing or validation fails, a safe text message is returned and no tool is executed
- Tool results larger than `ADAPTER_MAX_TOOL_RESULT_CHARS` are truncated before the next Ollama call

---

## Security controls

- **Loopback only.** `ADAPTER_HOST` must be `127.0.0.1`, `::1`, or `localhost`.
- **Tool allow-list per request.** Only tools advertised by OpenCode in the current request can be returned. Unknown tools are rejected.
- **Schema validation.** Normalized arguments must satisfy the advertised JSON Schema.
- **Workspace confinement.** File paths must resolve inside `ADAPTER_WORKSPACE`. Traversal (`..`), absolute paths outside the root, UNC/device paths, drive-relative paths, wildcards, NUL bytes, reserved Windows names, and drive/stream separators are rejected.
- **Shell guard.** Obvious destructive commands are blocked (recursive deletes, disk format, shutdown, service/task modification, encoded PowerShell, download-and-execute pipelines, destructive git operations, and similar). Background commands are disabled. `timeout` is clamped to `ADAPTER_MAX_COMMAND_TIMEOUT_MS`, and commands have a length limit.
- **Size and count limits.** Request/response size, tool-call count, tool-call cycles, and repeated calls are bounded. Large tool results (file reads, command output) are truncated before being sent back to Ollama, protecting the context window.
- **Optional bearer auth.** When `ADAPTER_API_KEY` is set, `/v1/*` requires `Authorization: Bearer <token>`.
- **Logging hygiene.** Prompts, file contents, tool arguments, and request bodies are not logged. Bearer tokens are redacted. `httpx`/`httpcore` URL logging is quieted.
- **No arbitrary execution by the adapter.** The adapter never executes model-generated shell commands or Python code itself. It only returns validated tool calls to OpenCode.

**Important limitation:** the command denylist is defense-in-depth, not a sandbox. A determined model can construct a command that bypasses pattern checks, and OpenCode executes tools with your user privileges. Use OpenCode permissions and a low-privilege account/container when running untrusted models. The workspace guard covers the file paths and shell arguments the adapter knows about; it is not a general OS sandbox.

---

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Current result on this machine:

```text
167 passed, 1 skipped in 0.95s
```

The skipped test is a symlink-escape test that requires Windows symlink privileges (Developer Mode or an elevated shell). The other path-escape, schema, parser, SSE, auth, limit, and security tests pass.

---

## Acceptance evidence

A real OpenCode run was performed against Ollama with `qwen2.5-coder:7b`, not a mock. Summary:

| Check | Result |
|---|---|
| Adapter health | `status=ok`, Ollama `0.40.11` reachable |
| OpenCode model list | `ollama-adapter/qwen2.5-coder:7b` listed |
| Read | `read` executed; returned actual `input.txt` contents |
| Write | `write` executed; created `result.txt` |
| Edit | `edit` executed; changed `hello adapter` to `hello adapter v2` |
| Shell | `shell` executed `python -V`; exit 0; returned `Python 3.10.0` |
| Compile check | `python -m py_compile check.py`; exit 0; `.pyc` created |
| Global config | Provider loaded from global config; read run succeeded |

Detailed commands and captured output are in [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md).

A useful safety observation from the acceptance run: after `python -m py_compile check.py` succeeded, the model repeated the same shell call. The adapter blocked the duplicate with its safe-repeat message instead of running the command twice. A second, clearer run (`python -V`) completed normally and the model reported the output.

---

## Revert OpenCode configuration

List available backups:

```powershell
.\scripts\revert-opencode-config.ps1 -List
```

Restore a specific backup:

```powershell
.\scripts\revert-opencode-config.ps1 -BackupPath "C:\Users\<you>\.config\opencode\opencode.json.bak-20261027-073223"
```

Restore the newest backup:

```powershell
.\scripts\revert-opencode-config.ps1
```

Restart OpenCode after restoring. The backup made during this setup is:

```text
C:\Users\lihin\.config\opencode\opencode.json.bak-20261027-073223
```

The installer never deletes the existing `providers.ollama` block. If you prefer to remove only the adapter manually, delete the `provider.ollama-adapter` object and restart OpenCode.

---

## Troubleshooting

### `Ollama is not reachable`

```powershell
ollama list
Invoke-RestMethod http://127.0.0.1:11434/api/version
```

Start Ollama if needed. Check `OLLAMA_BASE_URL` if Ollama runs on a different port.

### `Model is not in ADAPTER_MODELS`

Add the model to `ADAPTER_MODELS` (comma-separated) and restart the adapter. The default allow-list is only `qwen2.5-coder:7b`.

### OpenCode does not list `ollama-adapter`

```powershell
opencode debug config
opencode models
```

Confirm the `provider.ollama-adapter` block exists and `baseURL` ends in `/v1`. Restart OpenCode or use `opencode run --standalone` to avoid a stale background server.

### Adapter starts, but OpenCode gets connection errors

- Is the adapter still running on port 8765?
- Is `baseURL` exactly `http://127.0.0.1:8765/v1`?
- Is another process using port 8765?

```powershell
Get-NetTCPConnection -LocalPort 8765 -State Listen
```

### Tool call is rejected with a safe message

The adapter rejected the model output rather than executing something unsafe or invalid. Common causes:

- the model named a tool that was not advertised in that request
- malformed JSON or missing required arguments
- a path outside `ADAPTER_WORKSPACE`
- a blocked shell command
- a duplicate call

Retry with a clearer prompt. Inspect adapter stderr for the rejection category (the adapter does not log prompts or file contents).

### `401 Invalid or missing API key`

Set the same token in `ADAPTER_API_KEY` and in OpenCode's provider `options.apiKey`, then restart both.

### `ADAPTER_WORKSPACE` does not match the project

The adapter rejects paths outside its configured workspace even if OpenCode is running elsewhere. Restart the adapter with the correct `-Workspace` path.

### Port already in use

```powershell
Get-NetTCPConnection -LocalPort 8765 -State Listen | Select-Object LocalAddress,LocalPort,OwningProcess
```

Stop the old adapter or start the new one with `-Port 8766` and update `baseURL`.

### The model's final answer is a safe-repeat message

Small models sometimes repeat a tool call after seeing the result. The adapter blocks the duplicate. Ask a follow-up question or use a more explicit prompt; the original tool result is still in the conversation.

### PowerShell execution policy

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

This affects only the current PowerShell session.

---

## Project layout

```text
src/adapter/
  main.py             FastAPI app, chat service, SSE, auth, body limit
  protocol.py         OpenAI/Ollama models, tool registry, aliases, schema validation
  tool_parser.py      structured, tagged, and bare-JSON tool-call parsing
  security.py         workspace path guard and destructive-command checks
  ollama_client.py    Ollama native /api/chat client and error mapping
  config.py           environment settings and validation
  logging_config.py   logging setup and secret redaction
  errors.py           OpenAI-compatible error types
tests/                parser, protocol, security, API, config, client tests
scripts/
  setup.ps1                     create venv, install, run tests
  start-adapter.ps1             start the adapter for a workspace
  install-opencode-provider.ps1 back up and add the OpenCode provider
  revert-opencode-config.ps1    restore a config backup
examples/opencode.json            sample provider configuration
docs/ACCEPTANCE.md                real end-to-end acceptance evidence
```

---

## Known limitations

- **Small-model quality.** Qwen2.5-Coder 7B sometimes emits malformed JSON, repeats a call, or ignores a requested tool. The adapter rejects bad calls safely; it cannot make the model smarter.
- **Buffered responses.** Ollama is called with `stream: false`, so the adapter cannot stream tokens as they are generated. It emits a valid SSE stream after the full response arrives.
- **Shell output is controlled by OpenCode.** The adapter cannot intercept the shell's stdout/stderr directly; it truncates the tool result before the next Ollama call. OpenCode's shell tool also applies its own output handling.
- **Not a sandbox.** The workspace guard and command denylist are guard rails, not OS isolation.
- **One workspace per process.** The adapter's allowed root is fixed at startup.
- **Deprecated config key.** The adapter uses OpenCode's `provider` (singular) compatibility key. OpenCode 2.1.4 accepts it and normalizes it to `providers` internally. A future OpenCode version may require migrating the block to the newer `providers` shape.
- **Tool-result formatting depends on the model.** Ollama's native `role: tool` messages are forwarded as-is; some models may not use them well.

---

## License

MIT. See [`LICENSE`](LICENSE).
