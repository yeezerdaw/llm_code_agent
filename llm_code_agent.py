#!/usr/bin/env python3
"""
llm_code_agent.py — Claude Code-style local agent (v2)
LLM    : Ollama  (default model: gemma4:e2b)
Sandbox: Local filesystem + subprocess (sandboxed working directory)

Features (v2):
  1. Planner-Executor-Observer state machine
  2. Parallel tool execution via ThreadPoolExecutor
  3. Self-healing tool calls with exponential backoff
  4. Turn limiter (--max-turns, default 15)
  5. Approval mode (--approval) for destructive commands
  6. Rich CLI with spinners, syntax highlighting, tool-call trees
  7. Conversation summarisation every N turns
  8. Persistent memory (remember / recall tools)
  9. /selfreview slash command
 10. Auto-test hook (--auto-test)

Setup:
    pip install requests rich
    ollama pull gemma4:e2b

Usage:
    python llm_code_agent.py
    python llm_code_agent.py --model gemma4:e2b --approval --auto-test
    python llm_code_agent.py --model qwen2.5:7b --max-turns 20
    python llm_code_agent.py --workdir /tmp/workspace --no-rich
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── OPTIONAL DEPENDENCY: rich ────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text
    from rich.markdown import Markdown
    from rich import box as rich_box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ── CONSTANTS ────────────────────────────────────────────────────────────────
OLLAMA_BASE = "http://localhost:11434"
MEMORY_FILE = "memory.json"
MAX_TURNS_DEFAULT = 15
SUMMARISE_EVERY = 5
CONTEXT_CHAR_LIMIT = 2000
DISPLAY_CHAR_LIMIT = 500

DANGEROUS_PATTERNS = re.compile(
    r'\b(rm|rmdir|mv|sudo|chmod|chown|mkfs|dd|shutdown|reboot|kill|pkill)\b'
)

LEXER_MAP = {
    '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
    '.html': 'html', '.css': 'css', '.json': 'json', '.md': 'markdown',
    '.sh': 'bash', '.yml': 'yaml', '.yaml': 'yaml', '.rs': 'rust',
    '.go': 'go', '.java': 'java', '.c': 'c', '.cpp': 'cpp',
    '.rb': 'ruby', '.sql': 'sql', '.xml': 'xml', '.toml': 'toml',
    '.txt': 'text', '.cfg': 'ini', '.ini': 'ini', '.env': 'bash',
}


# ══════════════════════════════════════════════════════════════════════════════
# §1  DISPLAY LAYER — Rich CLI with graceful ANSI fallback
# ══════════════════════════════════════════════════════════════════════════════

class Display:
    """
    Abstraction over terminal output.
    Uses the `rich` library for spinners, panels, syntax highlighting, and
    coloured trees when available.  Falls back to ANSI escape codes otherwise.
    """

    # ANSI codes (fallback)
    _RESET   = "\033[0m"
    _BOLD    = "\033[1m"
    _DIM     = "\033[2m"
    _CYAN    = "\033[96m"
    _GREEN   = "\033[92m"
    _YELLOW  = "\033[93m"
    _RED     = "\033[91m"
    _BLUE    = "\033[94m"
    _MAGENTA = "\033[95m"

    def __init__(self, use_rich: bool = False):
        self.use_rich = use_rich
        self._status = None
        if use_rich:
            self.console = Console()

    # ── helpers ──────────────────────────────────────────────────────────
    def _ansi(self, text, *codes):
        return f"{''.join(codes)}{text}{self._RESET}"

    @staticmethod
    def _guess_lexer(path: str) -> str:
        _, ext = os.path.splitext(path)
        return LEXER_MAP.get(ext.lower(), 'text')

    # ── banner ───────────────────────────────────────────────────────────
    def banner(self, model: str):
        if self.use_rich:
            title = Text.assemble(
                ("LLM Code Agent ", "bold cyan"),
                ("v2", "bold white"),
                ("  ·  ", "dim"),
                (f"{model}  (local)", "dim"),
            )
            self.console.print()
            self.console.print(Panel(
                title, box=rich_box.DOUBLE, border_style="cyan", padding=(0, 2),
            ))
            cmds = Text.assemble(
                ("Commands: ", "bold"),
                ("/exit ", "yellow"), (" ", ""), ("/clear ", "yellow"), (" ", ""),
                ("/model <n> ", "yellow"), (" ", ""), ("/workdir ", "yellow"),
                (" ", ""), ("/selfreview ", "yellow"), (" ", ""), ("/memory", "yellow"),
            )
            self.console.print(cmds)
            self.console.print()
        else:
            inner = f"  LLM Code Agent v2  ·  {model}  (local)"
            w = max(45, len(inner) + 2)
            pad = w - len(inner)
            bar = "═" * w
            print(f"\n{self._ansi('╔' + bar + '╗', self._CYAN)}")
            print(
                f"{self._ansi('║', self._CYAN)}  "
                f"{self._ansi('LLM Code Agent v2', self._BOLD, self._CYAN)}  ·  "
                f"{self._ansi(f'{model}  (local)', self._DIM)}"
                f"{' ' * pad}{self._ansi('║', self._CYAN)}"
            )
            print(f"{self._ansi('╚' + bar + '╝', self._CYAN)}")
            print(
                f"{self._ansi('Commands:', self._BOLD)}  "
                f"{self._ansi('/exit', self._YELLOW)}  "
                f"{self._ansi('/clear', self._YELLOW)}  "
                f"{self._ansi('/model <n>', self._YELLOW)}  "
                f"{self._ansi('/workdir', self._YELLOW)}  "
                f"{self._ansi('/selfreview', self._YELLOW)}  "
                f"{self._ansi('/memory', self._YELLOW)}"
            )
            print()

    # ── spinner ──────────────────────────────────────────────────────────
    def thinking_start(self, label: str = "Thinking"):
        if self.use_rich:
            self._status = self.console.status(
                f"[dim]{label}…[/dim]", spinner="dots"
            )
            self._status.start()
        else:
            print(self._ansi(f"  {label}…", self._DIM), end="\r", flush=True)

    def thinking_stop(self):
        if self.use_rich:
            if self._status:
                self._status.stop()
                self._status = None
        else:
            print("                              ", end="\r")

    # ── assistant text ───────────────────────────────────────────────────
    def assistant(self, text: str):
        if self.use_rich:
            self.console.print()
            self.console.print(
                Text("Assistant", style="bold cyan"), end=": "
            )
            try:
                self.console.print(Markdown(text))
            except Exception:
                self.console.print(text)
            self.console.print()
        else:
            print(f"\n{self._ansi('Assistant', self._CYAN, self._BOLD)}: {text}\n")

    # ── tool call ────────────────────────────────────────────────────────
    def tool_call(self, name: str, args: dict):
        preview = json.dumps(args, ensure_ascii=False)
        if len(preview) > 120:
            preview = preview[:117] + "…"
        if self.use_rich:
            self.console.print(
                f"  [yellow]⚙[/yellow]  [bold]{name}[/bold]"
                f"([dim]{preview}[/dim])"
            )
        else:
            print(
                f"  {self._ansi('⚙', self._YELLOW)}  "
                f"{self._ansi(name, self._BOLD)}"
                f"({self._ansi(preview, self._DIM)})"
            )

    # ── tool result ──────────────────────────────────────────────────────
    def tool_result(self, result: str, tool_name: str = "",
                    tool_args: dict = None, is_error: bool = False,
                    healed: bool = False):
        if is_error:
            prefix, style = "✗", "red"
        elif healed:
            prefix, style = "⟲", "yellow"
        else:
            prefix, style = "→", "green"

        truncated = (len(result) > DISPLAY_CHAR_LIMIT)
        display = result[:DISPLAY_CHAR_LIMIT] + ("\n…(truncated)" if truncated else "")

        # Syntax-highlighted file read
        if (self.use_rich and tool_name == "read_file"
                and tool_args and not result.startswith("ERROR:")):
            path = tool_args.get("path", "")
            lexer = self._guess_lexer(path)
            self.console.print(f"  [{style}]{prefix}[/{style}] ", end="")
            try:
                syn = Syntax(display, lexer, theme="monokai",
                             line_numbers=True, word_wrap=True)
                self.console.print(syn)
            except Exception:
                self.console.print(display)
            self.console.print()
        elif self.use_rich:
            self.console.print(f"  [{style}]{prefix}[/{style}] {display}\n")
        else:
            ansi_map = {"red": self._RED, "yellow": self._YELLOW, "green": self._GREEN}
            print(f"  {self._ansi(prefix, ansi_map[style])} {display}\n")

    # ── status / phase / observer ────────────────────────────────────────
    def phase(self, name: str, turn: int, max_turns: int):
        msg = f"─── {name} (turn {turn}/{max_turns}) ───"
        if self.use_rich:
            self.console.print(f"  [dim]{msg}[/dim]")
        else:
            print(self._ansi(f"  {msg}", self._DIM))

    def observer(self, text: str):
        if self.use_rich:
            self.console.print(f"  [magenta]👁 {text}[/magenta]")
        else:
            print(self._ansi(f"  👁 {text}", self._MAGENTA))

    def info(self, text: str):
        if self.use_rich:
            self.console.print(f"  [blue]ℹ[/blue] {text}")
        else:
            print(self._ansi(f"  ℹ {text}", self._BLUE))

    def warning(self, text: str):
        if self.use_rich:
            self.console.print(f"  [yellow]⚠ {text}[/yellow]")
        else:
            print(self._ansi(f"  ⚠ {text}", self._YELLOW))

    def error(self, text: str):
        if self.use_rich:
            self.console.print(f"  [red]✗ {text}[/red]")
        else:
            print(self._ansi(f"  ✗ {text}", self._RED))

    def success(self, text: str):
        if self.use_rich:
            self.console.print(f"  [green]✓ {text}[/green]")
        else:
            print(self._ansi(f"  ✓ {text}", self._GREEN))

    # ── user prompt ──────────────────────────────────────────────────────
    def prompt(self) -> str | None:
        try:
            if self.use_rich:
                return self.console.input("[bold blue]You[/bold blue] › ").strip()
            else:
                return input(f"{self._ansi('You', self._BLUE, self._BOLD)} › ").strip()
        except (EOFError, KeyboardInterrupt):
            return None


# Global display instance — initialised in main()
ui: Display = Display(use_rich=False)


# ══════════════════════════════════════════════════════════════════════════════
# §2  TOOL DEFINITIONS (OpenAI-style JSON schema for Ollama)
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from the working directory."
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
            "description": (
                "Write (or overwrite) a file. Parent directories are created "
                "automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to write to."
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content to write."
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": (
                "Replace a specific string in a file with new content. "
                "Use this for targeted edits instead of rewriting the whole file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "old_content": {
                        "type": "string",
                        "description": "Exact string to replace (must match)."
                    },
                    "new_content": {
                        "type": "string",
                        "description": "Replacement string."
                    },
                },
                "required": ["path", "old_content", "new_content"],
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
                        "description": "Directory to list. Defaults to '.'."
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
                "Run a shell command in the working directory. "
                "Python 3, pip, git, and standard Unix tools are available."
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
    # ── Feature §8: Persistent Memory tools ──────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save a fact or user preference to persistent memory. "
                "Persists across sessions. Example keys: "
                "'preferred_test_framework', 'coding_style'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Short descriptive key."
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to remember."
                    },
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Retrieve a fact from persistent memory. "
                "Pass empty key to list all stored memories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key to look up (empty = list all)."
                    }
                },
                "required": [],
            },
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# §3  LOCAL TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def resolve(path: str, workdir: str) -> str | None:
    """Resolve a path relative to workdir, blocking directory traversal."""
    full = os.path.realpath(os.path.join(workdir, path))
    if not full.startswith(os.path.realpath(workdir)):
        return None
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
        d = os.path.dirname(full)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} bytes → {full}"
    except Exception as e:
        return f"ERROR: {e}"


def tool_replace_in_file(path: str, old_content: str,
                         new_content: str, workdir: str) -> str:
    full = resolve(path, workdir)
    if full is None:
        return "ERROR: Path traversal outside working directory is not allowed."
    try:
        with open(full, "r", encoding="utf-8") as f:
            content = f.read()
        if old_content not in content:
            return (
                "ERROR: old_content not found in the file. "
                "Make sure you matched the existing string exactly."
            )
        content = content.replace(old_content, new_content, 1)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully replaced content in {full}"
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
    env = os.environ.copy()
    venv_bin = os.path.join(workdir, ".venv", "bin")
    if os.path.isdir(venv_bin):
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = os.path.join(workdir, ".venv")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True,
            text=True, timeout=timeout, cwd=workdir, env=env,
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


# ── §8: Persistent Memory implementation ─────────────────────────────────────

def load_memory(workdir: str) -> dict:
    path = os.path.join(workdir, MEMORY_FILE)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_memory(workdir: str, memory: dict):
    path = os.path.join(workdir, MEMORY_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def tool_remember(key: str, value: str, workdir: str) -> str:
    memory = load_memory(workdir)
    memory[key] = value
    save_memory(workdir, memory)
    return f"Remembered: {key} = {value}"


def tool_recall(key: str, workdir: str) -> str:
    memory = load_memory(workdir)
    if not key:
        if not memory:
            return "No memories stored yet."
        return json.dumps(memory, indent=2, ensure_ascii=False)
    return memory.get(key, f"No memory found for key: '{key}'")


# ── Tool dispatcher ──────────────────────────────────────────────────────────

def dispatch_tool(name: str, args: dict, workdir: str) -> str:
    """Route a tool call to its implementation."""
    if name == "read_file":
        return tool_read_file(args["path"], workdir)
    elif name == "write_file":
        return tool_write_file(args["path"], args["content"], workdir)
    elif name == "replace_in_file":
        return tool_replace_in_file(
            args["path"], args["old_content"], args["new_content"], workdir
        )
    elif name == "list_files":
        return tool_list_files(args.get("path", "."), workdir)
    elif name == "run_command":
        return tool_run_command(args["command"], args.get("timeout", 60), workdir)
    elif name == "remember":
        return tool_remember(args["key"], args["value"], workdir)
    elif name == "recall":
        return tool_recall(args.get("key", ""), workdir)
    return f"ERROR: Unknown tool '{name}'"


# ══════════════════════════════════════════════════════════════════════════════
# §3b  SELF-HEALING TOOL DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

def self_heal_dispatch(name: str, args: dict, workdir: str,
                       max_retries: int = 3) -> tuple[str, bool]:
    """
    Execute a tool call with automatic recovery:
      - File not found  → search for similar filenames and retry
      - Transient error → exponential backoff retry
    Returns (result_string, was_healed_bool).
    """
    result = dispatch_tool(name, args, workdir)

    if not result.startswith("ERROR:"):
        return result, False

    # ── Heal: File not found → fuzzy search ──────────────────────────────
    if ("File not found" in result
            and name in ("read_file", "replace_in_file")):
        basename = os.path.basename(args.get("path", ""))
        if basename:
            search_result = tool_run_command(
                f"find . -name '*{basename}*' -type f 2>/dev/null | head -5",
                10, workdir,
            )
            # Parse find output
            candidates = []
            for line in search_result.split("\n"):
                line = line.strip()
                if line.startswith("STDOUT:"):
                    continue
                if line.startswith("Exit code:"):
                    continue
                if line.startswith("STDERR:"):
                    continue
                if line and not line.startswith("ERROR:"):
                    candidates.append(line)
            if candidates:
                args_copy = dict(args)
                args_copy["path"] = candidates[0]
                retry = dispatch_tool(name, args_copy, workdir)
                if not retry.startswith("ERROR:"):
                    return (
                        f"[Self-healed: used '{candidates[0]}' "
                        f"instead of '{args['path']}']\n{retry}"
                    ), True

    # ── Heal: Transient errors → exponential backoff ─────────────────────
    if name == "run_command" and (
        "timed out" in result.lower()
        or "connection" in result.lower()
        or "temporary" in result.lower()
    ):
        for attempt in range(1, max_retries + 1):
            delay = 2 ** (attempt - 1)  # 1s, 2s, 4s
            time.sleep(delay)
            result = dispatch_tool(name, args, workdir)
            if not result.startswith("ERROR:"):
                return (
                    f"[Self-healed: succeeded on retry {attempt} "
                    f"after {delay}s]\n{result}"
                ), True

    return result, False


# ══════════════════════════════════════════════════════════════════════════════
# §5  APPROVAL MODE — gate destructive operations
# ══════════════════════════════════════════════════════════════════════════════

def check_approval(name: str, args: dict, workdir: str,
                   approval_enabled: bool) -> tuple[bool, str]:
    """
    Returns (needs_approval: bool, warning_message: str).
    Only active when --approval flag is set.
    """
    if not approval_enabled:
        return False, ""

    if name == "run_command":
        cmd = args.get("command", "")
        if DANGEROUS_PATTERNS.search(cmd):
            return True, f"This command may modify the system:\n    {cmd}"

    if name == "write_file":
        full = resolve(args.get("path", ""), workdir)
        if full and os.path.exists(full):
            return True, f"File already exists and will be overwritten:\n    {full}"

    return False, ""


def prompt_user_approval(message: str) -> bool:
    """Prompt the user for y/N confirmation."""
    ui.warning(message)
    try:
        answer = input("  Proceed? (y/N) › ").strip().lower()
        return answer == "y"
    except (EOFError, KeyboardInterrupt):
        return False


# ══════════════════════════════════════════════════════════════════════════════
# §6  OLLAMA API
# ══════════════════════════════════════════════════════════════════════════════

def chat_ollama(model: str, messages: list, use_tools: bool = True) -> dict:
    """
    Send a chat request to the local Ollama instance.
    Set use_tools=False for summarisation / self-review (no tool calling).
    """
    import requests

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if use_tools:
        payload["tools"] = TOOLS

    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/chat", json=payload, timeout=300
        )
        r.raise_for_status()
        return r.json()["message"]
    except requests.exceptions.ConnectionError:
        ui.error("Cannot reach Ollama. Run:  ollama serve")
        sys.exit(1)
    except Exception as e:
        ui.error(f"Ollama error: {e}")
        sys.exit(1)


def list_ollama_models() -> list:
    import requests
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# §7  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert coding assistant running locally on the user's machine.
You have a sandboxed working directory where you can read/write files and run
shell commands.

Available tools:
  read_file       — Read a file's contents.
  write_file      — Write or overwrite a file (parent dirs auto-created).
  replace_in_file — Surgically replace a specific string in a file.
  list_files      — List files and directories at a given path.
  run_command     — Execute shell commands (Python 3, pip, git, unix tools).
  remember        — Save a fact/preference to persistent memory (across sessions).
  recall          — Retrieve a fact from memory (empty key = list all).

Rules:
1. Think step-by-step before acting.
2. Always use tools — never just describe what to do.
3. Prefer replace_in_file over write_file for edits to existing files.
4. Read command output and fix errors automatically — do not give up easily.
5. Stay within the working directory.
6. Never ask the user to run commands manually.
7. Use remember to store user preferences you discover.
8. When writing Python files, consider whether a test file is appropriate.
"""


# ══════════════════════════════════════════════════════════════════════════════
# §7b  CONVERSATION SUMMARISATION
# ══════════════════════════════════════════════════════════════════════════════

def maybe_summarise(model: str, messages: list, turn: int) -> list:
    """
    Every SUMMARISE_EVERY turns, compress the conversation history by asking
    the LLM to summarise it, keeping the system prompt + summary + last 2
    messages.  This prevents context-window overflow on long sessions.
    """
    if turn % SUMMARISE_EVERY != 0 or turn == 0:
        return messages
    if len(messages) < 8:
        return messages  # Not enough history to warrant summarisation

    # Build the content to summarise (skip system prompt, keep last 2)
    to_summarise = []
    for m in messages[1:-2]:
        role = m.get("role", "?")
        content = (m.get("content") or "")[:300]
        to_summarise.append(f"[{role}]: {content}")

    summary_text_input = "\n".join(to_summarise)[:3000]

    summary_request = [
        {
            "role": "system",
            "content": (
                "Summarise the following conversation in 3-5 sentences. "
                "Focus on: what the user asked, what files were created or "
                "modified, what worked, and what failed. Be concise."
            ),
        },
        {"role": "user", "content": summary_text_input},
    ]

    try:
        ui.info("Summarising conversation history…")
        response = chat_ollama(model, summary_request, use_tools=False)
        summary = (response.get("content") or "").strip()
        if summary:
            new_messages = [
                messages[0],  # system prompt
                {
                    "role": "system",
                    "content": f"[Conversation summary (turns 1-{turn})]: {summary}",
                },
                *messages[-2:],  # preserve last 2 messages
            ]
            ui.success(f"Compressed {len(messages)} messages → {len(new_messages)}")
            return new_messages
    except Exception as e:
        ui.warning(f"Summarisation failed (non-fatal): {e}")

    return messages


# ══════════════════════════════════════════════════════════════════════════════
# §10  AUTO-TEST HOOK
# ══════════════════════════════════════════════════════════════════════════════

def maybe_suggest_test(tool_name: str, tool_args: dict,
                       auto_test: bool) -> str | None:
    """
    If --auto-test is enabled and the agent just wrote a Python file,
    return a nudge message for the observer to inject, prompting the model
    to write and run tests.
    """
    if not auto_test:
        return None
    if tool_name != "write_file":
        return None
    path = tool_args.get("path", "")
    if not path.endswith(".py"):
        return None
    basename = os.path.basename(path)
    if basename.startswith("test_") or basename.endswith("_test.py"):
        return None  # Already a test file
    test_name = f"test_{basename}"
    return (
        f"[Auto-test] You wrote {path}. Write a unit test file "
        f"({test_name}) and run it with: python -m pytest {test_name} -v"
    )


# ══════════════════════════════════════════════════════════════════════════════
# §1  STATE MACHINE — Planner → Executor → Observer agentic loop
#     Also implements: §2 parallel execution, §3 self-healing,
#     §4 turn limiter, §5 approval gating, §10 auto-test hook
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(model: str, user_input: str, messages: list,
              workdir: str, config) -> list:
    """
    Main agentic loop structured as a 3-phase state machine:

      PLAN    — Send messages to LLM, receive reasoning + tool calls.
      EXECUTE — Dispatch tool calls (parallel if multiple), with approval
                gating and self-healing.
      OBSERVE — Analyse results, inject observer notes, trigger
                summarisation, loop back to PLAN.

    Exits when the model responds with text only (no tool calls) or
    when the turn limiter is reached.
    """
    messages.append({"role": "user", "content": user_input})

    # Inject persistent memory context (update if already present)
    memory = load_memory(workdir)
    if memory:
        mem_msg = {
            "role": "system",
            "content": f"[Persistent memory]: {json.dumps(memory)}",
        }
        # Replace existing memory injection or insert after system prompt
        replaced = False
        for i, m in enumerate(messages):
            if (m.get("role") == "system"
                    and m.get("content", "").startswith("[Persistent memory]")):
                messages[i] = mem_msg
                replaced = True
                break
        if not replaced:
            messages.insert(1, mem_msg)

    for turn in range(1, config.max_turns + 1):

        # ── PHASE 1: PLAN ────────────────────────────────────────────────
        ui.phase("PLAN", turn, config.max_turns)
        ui.thinking_start(f"Planning (turn {turn})")
        response = chat_ollama(model, messages)
        ui.thinking_stop()

        tool_calls = response.get("tool_calls") or []
        text_content = (response.get("content") or "").strip()

        if text_content:
            ui.assistant(text_content)

        if not tool_calls:
            messages.append({"role": "assistant", "content": text_content})
            break  # ──── DONE ─────

        messages.append(response)

        # ── PHASE 2: EXECUTE ─────────────────────────────────────────────
        ui.phase("EXECUTE", turn, config.max_turns)

        # 2a. Parse tool calls
        tasks = []  # list of (fn_name, args_dict, pre_error_or_None)
        for tc in tool_calls:
            fn = tc["function"]["name"]
            args = tc["function"].get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    tasks.append((
                        fn, {},
                        f"ERROR: Malformed JSON arguments for tool {fn}. "
                        f"Fix your syntax before proceeding."
                    ))
                    continue
            tasks.append((fn, args, None))

        # 2b. Approval gate
        for i, (fn, args, err) in enumerate(tasks):
            if err is not None:
                continue
            needs, msg = check_approval(fn, args, workdir, config.approval)
            if needs:
                if not prompt_user_approval(msg):
                    tasks[i] = (fn, args, "BLOCKED: User denied execution.")

        # 2c. Split into executable vs blocked/errored
        to_execute = [(fn, args) for fn, args, err in tasks if err is None]
        blocked = [(fn, err) for fn, _, err in tasks if err is not None]

        all_results = []   # list of (fn_name, result_str, was_healed)
        test_nudges = []

        # Handle blocked
        for fn, err in blocked:
            ui.tool_call(fn, {})
            ui.tool_result(err, is_error=True)
            all_results.append((fn, err, False))

        # 2d. Execute — parallel if multiple, sequential otherwise
        if len(to_execute) > 1:
            ui.info(f"⚡ Dispatching {len(to_execute)} tools in parallel")
            with ThreadPoolExecutor(max_workers=4) as pool:
                future_map = {}
                for fn, args in to_execute:
                    ui.tool_call(fn, args)
                    fut = pool.submit(self_heal_dispatch, fn, args, workdir)
                    future_map[fut] = (fn, args)

                for fut in as_completed(future_map):
                    fn, args = future_map[fut]
                    try:
                        result, healed = fut.result()
                    except Exception as exc:
                        result, healed = f"ERROR: Thread exception: {exc}", False
                    ui.tool_result(
                        result, tool_name=fn, tool_args=args, healed=healed
                    )
                    all_results.append((fn, result, healed))
                    nudge = maybe_suggest_test(fn, args, config.auto_test)
                    if nudge:
                        test_nudges.append(nudge)
        else:
            for fn, args in to_execute:
                ui.tool_call(fn, args)
                result, healed = self_heal_dispatch(fn, args, workdir)
                ui.tool_result(
                    result, tool_name=fn, tool_args=args, healed=healed
                )
                all_results.append((fn, result, healed))
                nudge = maybe_suggest_test(fn, args, config.auto_test)
                if nudge:
                    test_nudges.append(nudge)

        # 2e. Append tool results to conversation
        for _fn, result, _healed in all_results:
            ctx = (
                result if len(result) <= CONTEXT_CHAR_LIMIT
                else result[:CONTEXT_CHAR_LIMIT]
                + "\n...(Output truncated. Use head/grep to see more)"
            )
            messages.append({"role": "tool", "content": ctx})

        # ── PHASE 3: OBSERVE ─────────────────────────────────────────────
        ui.phase("OBSERVE", turn, config.max_turns)

        errors = [
            r for _, r, _ in all_results
            if r.startswith("ERROR:") or r.startswith("BLOCKED:")
        ]
        heals = sum(1 for _, _, h in all_results if h)

        observer_parts = []
        if errors:
            summaries = [e[:80] for e in errors[:3]]
            observer_parts.append(
                f"{len(errors)} tool(s) failed: " + "; ".join(summaries)
            )
        if heals:
            observer_parts.append(f"{heals} error(s) were self-healed.")
        if test_nudges:
            observer_parts.extend(test_nudges)

        if observer_parts:
            obs = "[Observer] " + " | ".join(observer_parts) + " — Re-plan."
            messages.append({"role": "system", "content": obs})
            ui.observer(obs)
        else:
            ui.success("All tools executed successfully.")

        # Periodic summarisation
        messages = maybe_summarise(model, messages, turn)

    else:
        # ── TURN LIMIT REACHED ───────────────────────────────────────────
        ui.warning(
            f"Turn limit ({config.max_turns}) reached. "
            f"Requesting final answer…"
        )
        messages.append({
            "role": "system",
            "content": (
                "TURN LIMIT REACHED. Provide your best answer now based on "
                "what you have accomplished. Do NOT call any more tools."
            ),
        })
        ui.thinking_start("Final answer")
        response = chat_ollama(model, messages, use_tools=False)
        ui.thinking_stop()
        text = (response.get("content") or "").strip()
        if text:
            ui.assistant(text)
        messages.append({"role": "assistant", "content": text})

    return messages


# ══════════════════════════════════════════════════════════════════════════════
# §9  /selfreview — Meta-analysis of the agent's own code
# ══════════════════════════════════════════════════════════════════════════════

def run_selfreview(model: str):
    """Read the agent's own source and ask the LLM for a code review."""
    ui.info("Running self-review of agent code…")
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception:
        ui.error("Could not read own source code.")
        return

    # Truncate to fit reasonable context
    source_truncated = source[:8000]
    review_messages = [
        {
            "role": "system",
            "content": (
                "You are a senior Python code reviewer. Analyse the following "
                "code for bugs, anti-patterns, security issues, and improvement "
                "suggestions. Be specific. Cite line numbers. Format as a "
                "bulleted list grouped by severity (Critical / Warning / Info)."
            ),
        },
        {"role": "user", "content": source_truncated},
    ]

    ui.thinking_start("Reviewing")
    try:
        response = chat_ollama(model, review_messages, use_tools=False)
        ui.thinking_stop()
        text = (response.get("content") or "").strip()
        if text:
            ui.assistant(text)
        else:
            ui.warning("Model returned empty review.")
    except Exception as e:
        ui.thinking_stop()
        ui.error(f"Self-review failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# §12  MAIN / REPL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global ui

    parser = argparse.ArgumentParser(
        description="LLM Code Agent v2 — Claude Code-style local assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python llm_code_agent.py --model qwen2.5:7b --approval\n"
            "  python llm_code_agent.py --auto-test --max-turns 20\n"
            "  python llm_code_agent.py --no-rich --workdir ./my_project\n"
        ),
    )
    parser.add_argument(
        "--model", default="gemma4:e2b",
        help="Ollama model name (default: gemma4:e2b)",
    )
    parser.add_argument(
        "--workdir",
        default=os.path.join(os.getcwd(), "agent_workspace"),
        help="Working directory for file operations (default: ./agent_workspace)",
    )
    parser.add_argument(
        "--approval", action="store_true",
        help="Prompt before destructive commands (rm, mv, sudo, etc.)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=MAX_TURNS_DEFAULT,
        dest="max_turns",
        help=f"Max turns per agent invocation (default: {MAX_TURNS_DEFAULT})",
    )
    parser.add_argument(
        "--auto-test", action="store_true", dest="auto_test",
        help="Auto-suggest unit tests when writing Python files",
    )
    parser.add_argument(
        "--no-rich", action="store_true", dest="no_rich",
        help="Disable Rich output, use plain ANSI codes",
    )
    args = parser.parse_args()

    # Initialise display layer
    use_rich = HAS_RICH and not args.no_rich
    ui = Display(use_rich=use_rich)

    model = args.model
    workdir = os.path.realpath(args.workdir)
    os.makedirs(workdir, exist_ok=True)

    # Establish sterile field (venv)
    venv_path = os.path.join(workdir, ".venv")
    if not os.path.exists(venv_path):
        ui.info(f"Creating virtual environment in {workdir}…")
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", venv_path],
                check=True, capture_output=True,
            )
            ui.success("Virtual environment created.")
        except subprocess.CalledProcessError as e:
            err = (
                e.stderr.decode("utf-8", errors="ignore").strip()
                if e.stderr else "Unknown error"
            )
            ui.warning(f"Failed to create .venv: {err}")

    # Banner
    ui.banner(model)

    # Available models
    available = list_ollama_models()
    if available:
        ui.info(f"Ollama models: {', '.join(available)}")
    else:
        ui.warning("Could not list Ollama models — is Ollama running?")

    ui.info(f"Model  : {model}")
    ui.info(f"Workdir: {workdir}")
    if args.approval:
        ui.info("Approval mode: ON (destructive commands require confirmation)")
    if args.auto_test:
        ui.info("Auto-test: ON (will suggest tests for new Python files)")
    ui.info(f"Max turns: {args.max_turns}")
    print()

    # Load and display persistent memory
    memory = load_memory(workdir)
    if memory:
        ui.info(f"Loaded {len(memory)} memory entries from previous sessions.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ── REPL ─────────────────────────────────────────────────────────────
    while True:
        user_input = ui.prompt()
        if user_input is None:
            ui.info("Bye!")
            break

        if not user_input:
            continue

        # ── Slash commands ────────────────────────────────────────────────
        if user_input == "/exit":
            ui.info("Bye!")
            break

        if user_input == "/clear":
            messages = [messages[0]]
            # Re-inject memory
            mem = load_memory(workdir)
            if mem:
                messages.insert(1, {
                    "role": "system",
                    "content": f"[Persistent memory]: {json.dumps(mem)}",
                })
            ui.success("Session cleared. (Working directory and memory preserved)")
            continue

        if user_input == "/workdir":
            ui.info(f"Working directory: {workdir}")
            continue

        if user_input == "/memory":
            mem = load_memory(workdir)
            if mem:
                ui.info("Persistent memory:")
                for k, v in mem.items():
                    ui.info(f"  {k}: {v}")
            else:
                ui.info("No persistent memories stored.")
            continue

        if user_input.startswith("/model"):
            parts = user_input.split()
            if len(parts) == 2:
                model = parts[1]
                ui.success(f"Switched to: {model}")
            else:
                ui.info(f"Current model: {model}  |  /model <name>")
            continue

        if user_input == "/selfreview":
            run_selfreview(model)
            continue

        # ── Normal user message → agentic loop ───────────────────────────
        messages = run_agent(model, user_input, messages, workdir, args)


if __name__ == "__main__":
    main()
