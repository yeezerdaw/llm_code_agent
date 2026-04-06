# LLM Code Agent

A lightweight, terminal-based AI coding assistant inspired by Claude Code. It runs 100% locally on your machine using [Ollama](https://ollama.com/), providing an interactive REPL where the AI can read, write, and execute code within a safe workspace environment.

## 🌟 Features

- **100% Local Processing:** Privacy-first design using Ollama models (default is `gemma4:e2b`). No API keys or cloud subscriptions needed.
- **Agentic Loop:** The model natively understands how to pick tools, view error outputs, and self-correct across continuous loops.
- **Automated Tooling:** 
  - 📁 `list_files` — Inspect the local workspace.
  - 📖 `read_file` — Reads project files natively.
  - ✍️ `write_file` — Safely writes code or structure, creating directories automatically.
  - ⚡ `run_command` — Executes terminal commands, compiles code, runs scripts, or operates `git`.
- **Sandboxed Operations:** Prevents malicious directory traversal by enforcing restrictions out of the working space.

---

## 🚀 Getting Started

### Prerequisites
1. Install [Ollama](https://ollama.com/).
2. Make sure you have python 3 installed.

### Setup

1. Clone or download `llm_code_agent.py`.
2. Install the lightweight Python requirements:
```bash
pip install requests
```
3. Pull your preferred working model from Ollama:
```bash
ollama pull gemma4:e2b
```
*(You can also pull alternate models such as `qwen2.5:7b` depending on your local hardware).*

---

## 💻 Usage

Start the interactive console simply by running:
```bash
python llm_code_agent.py
```

### Command Line Flags

You can instantly override defaults by passing arguments natively:
- `--model`: Specific model version you want the engine to use (e.g. `--model qwen2.5:7b`).
- `--workdir`: Redefine the root working directory the agent is allowed to touch.

**Examples:**
```bash
# Use a custom model
python llm_code_agent.py --model llama3.2

# Set a custom isolated workspace
python llm_code_agent.py --workdir /path/to/my/sandbox
```

### In-Chat Slash Commands

Once the REPL has booted up, you can execute the following shortcuts:
- `/exit` — Quit the interface.
- `/clear` — Wipe the active memory payload to empty the context window limit (your written files remain intact).
- `/workdir` — Output the currently targeted active filesystem path.
- `/model <name>` — Live-swap the working LLM model gracefully.

---

## 🛡️ Safety & Security

By default, the script isolates and sandboxes target requests. Using `os.path.realpath`, it intercepts and disables tool calls attempting arbitrary external filesystem traversal. To keep your host reliable, standard user permissions apply to all commands executed by the LLM. 

Be cautious with destructive commands (e.g., `rm -rf`) in your workspace!
