# Acceptance evidence

Date: 2026-10-27

This is a real end-to-end test against the installed software, not a mock.

## Environment

| Component | Version / value |
|---|---|
| OpenCode | 2.1.4 |
| Ollama | 0.40.11 |
| Model | `qwen2.5-coder:7b` |
| Python | 3.10.0 |
| PowerShell | 5.1.26100.9711 |
| Adapter | `0.1.0` |
| Adapter URL | `http://127.0.0.1:8765` |
| Ollama URL | `http://127.0.0.1:11434` |
| Acceptance workspace | `C:\Users\lihin\AppData\Local\Temp\opencode\acceptance-project` |

The acceptance workspace contained:

```text
input.txt      "hello from opencode"
check.py       print("ok")
notes.md       short Markdown file
opencode.json  local provider config for the adapter
```

## 1. Adapter health

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
```

Captured response:

```json
{
  "status": "ok",
  "adapter": "ok",
  "version": "0.1.0",
  "ollama": {
    "reachable": true,
    "version": "0.40.11"
  },
  "models": ["qwen2.5-coder:7b"]
}
```

## 2. OpenAI-compatible model list

```powershell
Invoke-RestMethod http://127.0.0.1:8765/v1/models
```

Captured response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen2.5-coder:7b",
      "object": "model",
      "created": 0,
      "owned_by": "ollama-adapter"
    }
  ]
}
```

## 3. OpenCode provider visibility

```powershell
opencode models
```

Relevant output:

```text
ollama-adapter/qwen2.5-coder:7b
ollama/deepseek-coder:6.7b
ollama/qwen2.5-coder:7b
```

`opencode debug config` showed the adapter provider with tool capability enabled:

```json
{
  "providers": {
    "ollama-adapter": {
      "name": "Ollama Tool Adapter",
      "package": "aisdk:@ai-sdk/openai-compatible",
      "settings": {
        "timeout": 300000,
        "baseURL": "http://127.0.0.1:8765/v1"
      },
      "models": {
        "qwen2.5-coder:7b": {
          "name": "Qwen2.5 Coder 7B (adapter)",
          "capabilities": {
            "tools": true,
            "input": ["text", "image"],
            "output": ["text"]
          },
          "limit": {
            "context": 16384,
            "output": 4096
          }
        }
      }
    }
  }
}
```

The existing `providers.ollama` section remained present.

## 4. Real OpenCode tool executions

All runs used:

```powershell
opencode run --standalone --auto --format json -m ollama-adapter/qwen2.5-coder:7b "<prompt>"
```

### 4.1 Read

Prompt:

```text
Read the file input.txt and report its exact contents. Do not modify any files.
```

Captured `tool_use` event (trimmed):

```json
{
  "type": "tool_use",
  "part": {
    "type": "tool",
    "tool": "read",
    "state": {
      "status": "completed",
      "input": {"path": "input.txt"},
      "output": "Read file input.txt, lines 1-1\n1: hello from opencode\r"
    }
  }
}
```

The returned content matched the actual file. The model's final answer contained `hello from opencode` inside a Markdown code fence.

### 4.2 Write

Prompt:

```text
Create a new file named result.txt containing exactly the text 'hello adapter'.
Use the write tool. Do not modify any other file.
```

Captured `tool_use` event (trimmed):

```json
{
  "type": "tool_use",
  "part": {
    "type": "tool",
    "tool": "write",
    "state": {
      "status": "completed",
      "input": {"content": "hello adapter", "path": "result.txt"},
      "output": "Created file successfully: result.txt"
    }
  }
}
```

Verification:

```powershell
Get-Content .\result.txt -Raw
# hello adapter
```

### 4.3 Edit

Prompt:

```text
Edit result.txt and replace the exact text 'hello adapter' with 'hello adapter v2'.
Use the edit tool.
```

Captured `tool_use` event (trimmed):

```json
{
  "type": "tool_use",
  "part": {
    "type": "tool",
    "tool": "edit",
    "state": {
      "status": "completed",
      "input": {
        "newString": "hello adapter v2",
        "oldString": "hello adapter",
        "path": "result.txt"
      },
      "output": "Edited result.txt (1 replacement)"
    }
  }
}
```

Verification:

```powershell
Get-Content .\result.txt -Raw
# hello adapter v2
```

### 4.4 Shell

Prompt:

```text
Use the shell tool to run the command: python -V. Then report the version shown.
Run it only once.
```

Captured `tool_use` event (trimmed):

```json
{
  "type": "tool_use",
  "part": {
    "type": "tool",
    "tool": "shell",
    "state": {
      "status": "completed",
      "input": {"command": "python -V"},
      "output": "Python 3.10.0\r\n",
      "metadata": {
        "metadata": {
          "status": "completed",
          "truncated": false,
          "exit": 0
        }
      }
    }
  }
}
```

The model's final text was:

```text
The Python version shown is 3.10.0.
```

### 4.5 Python compile check

A separate run asked the model to execute:

```text
python -m py_compile check.py
```

The `shell` tool completed with `exit: 0` and produced:

```text
__pycache__/check.cpython-310.pyc
```

After the result, the model repeated the same shell call. The adapter blocked the duplicate and returned its safe-repeat message instead of running the command twice. This demonstrates the repeated-call guard in a real session. A clearer follow-up run (`python -V`, above) completed normally.

After the final code change (the tool-result truncation limit), the read and shell checks were repeated successfully: `read` returned `hello from opencode`, and `shell` ran `python -V` with `exit: 0` and output `Python 3.10.0`.

## 5. Global OpenCode config integration

The global config was backed up, then the adapter provider was added with:

```powershell
.\scripts\install-opencode-provider.ps1
```

Backup created:

```text
C:\Users\lihin\.config\opencode\opencode.json.bak-20261027-073223
```

The original `providers.ollama` block and the top-level default model were preserved. The local project config was temporarily renamed so the global config was the only source, and this run succeeded:

```powershell
opencode models
# ollama-adapter/qwen2.5-coder:7b

opencode run --standalone --auto --format json `
  -m ollama-adapter/qwen2.5-coder:7b `
  "Read input.txt and report its exact contents. Do not modify any files."
```

Captured event:

```json
{
  "type": "tool_use",
  "part": {
    "type": "tool",
    "tool": "read",
    "state": {
      "status": "completed",
      "input": {"path": "input.txt"},
      "output": "Read file input.txt, lines 1-1\n1: hello from opencode\r"
    }
  }
}
```

The final answer contained a Markdown code block with `hello from opencode`.

## 6. Automated tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Result:

```text
167 passed, 1 skipped in 0.95s
```

The skipped test is the symlink-escape test, which requires Windows symlink privileges (Developer Mode or an elevated shell).

## 7. Revert

List backups:

```powershell
.\scripts\revert-opencode-config.ps1 -List
```

Restore the backup from this run:

```powershell
.\scripts\revert-opencode-config.ps1 `
  -BackupPath "C:\Users\lihin\.config\opencode\opencode.json.bak-20261027-073223"
```

The restore script validates that the backup is JSON before copying it back. Restart OpenCode afterward.
