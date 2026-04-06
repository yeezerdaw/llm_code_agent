---

# LLM Code Agent (v2)

A production‑grade, terminal‑based AI coding assistant inspired by Claude Code. It runs **100% locally** using [Ollama](https://ollama.com/), with a powerful agentic loop that can read, write, execute, and even **test** code – all inside a safe, sandboxed workspace.

> **For a college project**: This agent demonstrates 10 advanced features including a state machine, parallel tool execution, self‑healing, approval mode, persistent memory, and a `/selfreview` meta‑analysis command.

---

## ✨ Key Features (v2)

| Feature | Description |
|---------|-------------|
| **Planner‑Executor‑Observer state machine** | The agent explicitly plans tool calls, executes them (in parallel when possible), then observes results and replans – mirroring human problem‑solving. |
| **Parallel tool execution** | Multiple independent tool calls (e.g., writing several files) run concurrently using `ThreadPoolExecutor`, speeding up multi‑step tasks. |
| **Self‑healing tool calls** | If a tool fails (e.g., file not found), the agent automatically searches for similar filenames or retries with exponential backoff. |
| **Turn limiter** | Prevents infinite loops – configurable with `--max-turns` (default 15). After the limit, the agent returns its best answer. |
| **Approval mode** | When `--approval` is enabled, dangerous commands (`rm`, `mv`, `sudo`, etc.) and overwriting existing files require user confirmation. |
| **Rich CLI output** | Optional `rich` library support for spinners, syntax‑highlighted code, tool‑call trees, and professional panels. Falls back to clean ANSI colours if `rich` is not installed. |
| **Conversation summarisation** | Every 5 turns (configurable), the agent compresses the chat history using the LLM itself, keeping context within limits for long sessions. |
| **Persistent memory** | The agent can `remember` facts and `recall` them across sessions. Memory is stored in `memory.json` inside the working directory. |
| **`/selfreview` command** | The agent reads its own source code and performs a code review, listing bugs, security issues, and improvement suggestions. |
| **Auto‑test hook** | When `--auto-test` is enabled, writing a `.py` file triggers a nudge to create and run unit tests – promoting test‑driven behaviour. |

---

## 🛠️ Tools Reference

The agent can use the following tools. It decides when to call them based on your request.

| Tool | Description | Example arguments |
|------|-------------|--------------------|
| `read_file` | Read the full content of a file. | `{"path": "script.py"}` |
| `write_file` | Write or overwrite a file (parent directories created automatically). | `{"path": "app.py", "content": "print('hello')"}` |
| `replace_in_file` | Replace the first occurrence of a string in a file – use for targeted edits. | `{"path": "app.py", "old_content": "hello", "new_content": "hi"}` |
| `list_files` | List files and directories in the workspace. | `{"path": "."}` (or omit) |
| `run_command` | Execute a shell command in the workspace. Supports timeouts. | `{"command": "python test.py", "timeout": 30}` |
| `remember` | Save a key‑value pair to persistent memory (cross‑session). | `{"key": "framework", "value": "pytest"}` |
| `recall` | Retrieve a value by key, or list all keys if empty. | `{"key": "framework"}` or `{}` |

---

## 💬 Slash Commands (REPL)

While the agent is running, type these commands at the `You ›` prompt:

| Command | Action |
|---------|--------|
| `/exit` | Quit the agent. |
| `/clear` | Clear the conversation history (keeps memory and workspace files). |
| `/workdir` | Show the current working directory (sandbox). |
| `/model <name>` | Switch to a different Ollama model live (e.g., `/model qwen2.5:14b`). |
| `/selfreview` | Ask the agent to analyse its own source code and report issues. |
| `/memory` | Display all persistent memories stored in `memory.json`. |

---

## 🚀 Getting Started

### Prerequisites

- **Ollama** installed and running ([download](https://ollama.com/))
- **Python 3.8+** (the script uses `venv` automatically)
- (Optional) `rich` library for enhanced terminal output

### Installation

1. **Clone or download** `llm_code_agent.py` into a folder of your choice.

2. **Install Python dependencies** (the agent will also attempt to create a virtual environment):
   ```bash
   pip install requests rich
   ```

3. **Pull an Ollama model** (recommended: `qwen2.5:7b` for good performance):
   ```bash
   ollama pull qwen2.5:7b
   ```

### Running the Agent

Basic usage (creates `./agent_workspace` as sandbox):
```bash
python llm_code_agent.py --model qwen2.5:7b
```

Advanced – enable approval mode and auto‑test, and set the workdir to the current folder:
```bash
python llm_code_agent.py --model qwen2.5:7b --workdir . --approval --auto-test
```

### Command‑line Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gemma4:e2b` | Ollama model name. |
| `--workdir` | `./agent_workspace` | Sandbox directory for file operations. |
| `--approval` | `False` | Ask for confirmation before dangerous commands or overwriting files. |
| `--max-turns` | `15` | Maximum agent turns per user request. |
| `--auto-test` | `False` | After writing a `.py` file, nudge the agent to write and run tests. |
| `--no-rich` | `False` | Disable `rich` output (use plain ANSI colours). |

---

## 🧪 Demo Examples

Here are three short interactions you can try to showcase the agent’s power.

### 1. Parallel file creation + auto‑test
```
You › Create two files: square.py with a function square(x) returning x*x, and cube.py with cube(x). Then write tests for both.
```
The agent will write all four files (two implementations, two tests) in parallel, run the tests, and report success.

### 2. Self‑healing edit
```
You › Write a file hello.py that prints "Hello world". Then replace "world" with "agent" and run it.
```
Even if the model makes a small mistake, the self‑healing layer can retry or adjust paths.

### 3. Persistent memory across sessions
```
You › Remember that I prefer pytest over unittest.
You › /exit
```
Restart the agent, then:
```
You › What is my preferred test framework?
```
The agent recalls `pytest` from `memory.json`.

### 4. Approval mode in action
```
You › Delete temp.txt
```
If `--approval` is on, the agent will ask: `⚠ This command may modify the system: rm temp.txt — Proceed? (y/N)`

---

## 🛡️ Safety & Security

- **Path sandboxing:** All file operations are resolved against the working directory. Any attempt to traverse outside (e.g., `../../etc/passwd`) is rejected.
- **Approval mode:** Destructive shell commands (`rm`, `mv`, `sudo`, etc.) and overwriting existing files require your consent when `--approval` is used.
- **Virtual environment:** The agent automatically creates a `.venv` inside the workdir and adds it to `PATH` for `run_command`, isolating package installations.
- **User responsibility:** Even with safety measures, avoid running the agent in sensitive directories. Review commands before approving.

---

## 🧠 Architecture Overview (Simplified)

```
User Input → [PLAN] → LLM generates tool calls
                ↓
         [EXECUTE] → Parallel dispatch (ThreadPoolExecutor)
                ↓
         [OBSERVE] → Analyse errors, inject observer notes, summarise history
                ↓
         Loop until final text response or turn limit
```

- **PLAN:** The LLM receives the conversation + memory and decides which tools to call.
- **EXECUTE:** Tools run concurrently. Self‑healing catches common errors. Approval mode gates dangerous actions.
- **OBSERVE:** The agent injects an observer message if any tool failed or was healed, then repeats the PLAN phase.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ConnectionError` to Ollama | Ensure Ollama is running: `ollama serve` |
| Model not found | Pull it first: `ollama pull <model>` |
| `rich` module not found | Install it: `pip install rich` or use `--no-rich` |
| Agent keeps calling tools without finishing | Increase `--max-turns` or use a larger model (e.g., `qwen2.5:14b`) |
| Permission errors when writing files | Check that the workdir is writable. Use `--workdir /tmp/agent_workspace` if needed. |

---

## 📄 License

This project is open source for educational purposes. Use it responsibly.

---
