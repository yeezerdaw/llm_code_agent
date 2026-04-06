#!/usr/bin/env python3
"""
llm_code_agent.py — Claude Code-style local agent
LLM    : Ollama  (default model: gemma4:e2b)
Sandbox: Local filesystem + subprocess (safe working directory)

Setup:
    pip install requests
    ollama pull gemma4:e2b

Usage:
    python llm_code_agent.py
    python llm_code_agent.py --model gemma4:e2b
    python llm_code_agent.py --model qwen2.5:7b
    python llm_code_agent.py --workdir /tmp/agent_workspace
"""

import argparse
import json
import os
import subprocess
import sys

# ── ANSI colours ─────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
BLUE   = "\033[94m"

def c(text, colour):
    return f"{colour}{text}{RESET}"

# ── TOOL DEFINITIONS (OpenAI-style JSON schema) ───────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the local working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file. Relative paths resolve from the working directory."
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (or overwrite) a file in the local working directory. Parent directories are created automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to write to. Relative paths resolve from the working directory."
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content to write to the file."
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories at a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list. Defaults to the working directory."
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command in the local working directory. "
                "Use for running scripts, installing packages (pip), "
                "running tests, compiling code, git operations, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 60)."
                    },
                },
                "required": ["command"],
            },
        },
    },
]

# ── LOCAL TOOL IMPLEMENTATIONS ────────────────────────────────────────────────

def resolve(path: str, workdir: str) -> str:
    """Resolve a path relative to workdir, preventing directory traversal."""
    full = os.path.realpath(os.path.join(workdir, path))
    if not full.startswith(os.path.realpath(workdir)):
        return None  # traversal attempt blocked
    return full


def tool_read_file(path: str, workdir: str) -> str:
    full = resolve(path, workdir)
    if full is None:
        return "ERROR: Path traversal outside working directory is not allowed."
    try:
        with open(full, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"ERROR: File not found: {path}"
    except Exception as e:
        return f"ERROR: {e}"


def tool_write_file(path: str, content: str, workdir: str) -> str:
    full = resolve(path, workdir)
    if full is None:
        return "ERROR: Path traversal outside working directory is not allowed."
    try:
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} bytes → {full}"
    except Exception as e:
        return f"ERROR: {e}"


def tool_list_files(path: str, workdir: str) -> str:
    target = resolve(path or ".", workdir)
    if target is None:
        return "ERROR: Path traversal outside working directory is not allowed."
    try:
        entries = os.listdir(target)
        if not entries:
            return "(empty directory)"
        lines = []
        for name in sorted(entries):
            full = os.path.join(target, name)
            size = os.path.getsize(full) if os.path.isfile(full) else "-"
            kind = "DIR" if os.path.isdir(full) else "FILE"
            lines.append(f"  {kind:<5}  {str(size):>8}  {name}")
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: {e}"


def tool_run_command(command: str, timeout: int, workdir: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )
        parts = []
        if result.stdout and result.stdout.strip():
            parts.append(f"STDOUT:\n{result.stdout.strip()}")
        if result.stderr and result.stderr.strip():
            parts.append(f"STDERR:\n{result.stderr.strip()}")
        parts.append(f"Exit code: {result.returncode}")
        return "\n".join(parts) if parts else "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"


def dispatch_tool(name: str, args: dict, workdir: str) -> str:
    if name == "read_file":
        return tool_read_file(args["path"], workdir)
    elif name == "write_file":
        return tool_write_file(args["path"], args["content"], workdir)
    elif name == "list_files":
        return tool_list_files(args.get("path", "."), workdir)
    elif name == "run_command":
        return tool_run_command(args["command"], args.get("timeout", 60), workdir)
    return f"ERROR: Unknown tool '{name}'"

# ── OLLAMA API ────────────────────────────────────────────────────────────────
OLLAMA_BASE = "http://localhost:11434"


def chat_ollama(model: str, messages: list) -> dict:
    import requests
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
    }
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]
    except requests.exceptions.ConnectionError:
        print(c("\n✗ Cannot reach Ollama. Run:  ollama serve", RED))
        sys.exit(1)
    except Exception as e:
        print(c(f"\n✗ Ollama error: {e}", RED))
        sys.exit(1)


def list_ollama_models() -> list:
    import requests
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert coding assistant running locally on the user's machine.
You have access to a working directory where you can freely read/write files
and run shell commands.

Rules:
- Think step-by-step before acting.
- Always use tools to create, inspect, run, and fix code — never just describe it.
- After running commands, read the output and fix errors automatically.
- Only operate within the designated working directory.
- Never ask the user to run commands themselves.
- Python 3, pip, and standard Unix tools are available.
"""

# ── AGENTIC LOOP ──────────────────────────────────────────────────────────────

def run_agent(model: str, user_input: str, messages: list, workdir: str) -> list:
    messages.append({"role": "user", "content": user_input})

    while True:
        print(c("  thinking…", DIM), end="\r", flush=True)
        response = chat_ollama(model, messages)
        print("              ", end="\r")

        tool_calls   = response.get("tool_calls") or []
        text_content = (response.get("content") or "").strip()

        if text_content:
            print(f"\n{c('Assistant', CYAN + BOLD)}: {text_content}\n")

        if not tool_calls:
            messages.append({"role": "assistant", "content": text_content})
            break

        messages.append(response)

        for tc in tool_calls:
            fn   = tc["function"]["name"]
            args = tc["function"].get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            preview = json.dumps(args, ensure_ascii=False)
            if len(preview) > 120:
                preview = preview[:117] + "…"
            print(f"  {c('⚙', YELLOW)}  {c(fn, BOLD)}({c(preview, DIM)})")

            result = dispatch_tool(fn, args, workdir)

            display = result if len(result) <= 500 else result[:500] + f"\n{c('…(truncated)', DIM)}"
            print(f"  {c('→', GREEN)} {display}\n")

            messages.append({"role": "tool", "content": result})

    return messages

# ── MAIN / REPL ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Local agentic coding assistant — gemma4:e2b")
    parser.add_argument("--model",   default="gemma4:e2b", help="Ollama model name (default: gemma4:e2b)")
    parser.add_argument("--workdir", default=os.path.join(os.getcwd(), "agent_workspace"),
                        help="Working directory for file operations (default: ./agent_workspace)")
    args = parser.parse_args()

    model   = args.model
    workdir = os.path.realpath(args.workdir)
    os.makedirs(workdir, exist_ok=True)

    banner = f"""
{c('╔═══════════════════════════════════════════╗', CYAN)}
{c('║', CYAN)}  {c('LLM Code Agent', BOLD + CYAN)}  ·  {c(f'{model}  (local)', DIM)}       {c('║', CYAN)}
{c('╚═══════════════════════════════════════════╝', CYAN)}
{c('Commands:', BOLD)}  {c('/exit', YELLOW)}  {c('/clear', YELLOW)}  {c('/model <n>', YELLOW)}  {c('/workdir', YELLOW)}
"""
    print(banner)

    available = list_ollama_models()
    if available:
        print(c("Ollama models available:", BOLD), ", ".join(available))
    else:
        print(c("⚠ Could not list Ollama models — is Ollama running?", YELLOW))

    print(c(f"\nModel  : {model}", GREEN))
    print(c(f"Workdir: {workdir}\n", GREEN))

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input(f"{c('You', BLUE + BOLD)} › ").strip()
        except (EOFError, KeyboardInterrupt):
            print(c("\nBye!", DIM))
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print(c("Bye!", DIM))
            break
        if user_input == "/clear":
            messages = [messages[0]]
            print(c("Session cleared.\n", DIM))
            continue
        if user_input == "/workdir":
            print(c(f"Working directory: {workdir}\n", CYAN))
            continue
        if user_input.startswith("/model"):
            parts = user_input.split()
            if len(parts) == 2:
                model = parts[1]
                print(c(f"Switched to: {model}\n", GREEN))
            else:
                print(c(f"Current model: {model}  |  /model <name>\n", YELLOW))
            continue

        messages = run_agent(model, user_input, messages, workdir)


if __name__ == "__main__":
    main()