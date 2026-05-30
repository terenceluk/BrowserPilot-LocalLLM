# 🌐 Browser Pilot — LM Studio

Control Chrome with natural language using a local LLM from LM Studio.

Built with [browser-use](https://github.com/browser-use/browser-use) + [Playwright](https://playwright.dev/) + [Gradio](https://gradio.app/).

## What It Does

You describe a task in natural language and the agent opens Chrome, navigates the web, clicks, types, and scrolls to complete it — then returns a summary of what it found. Everything runs locally; no cloud APIs or subscriptions required.

**Features:**
- **Natural language browser control** — describe what you want, the LLM plans the steps
- **Gradio web UI** with real-time step-by-step progress updates
- **Task queue** — submit multiple tasks and let the agent work through them sequentially
- **Task scheduling** — schedule tasks to run at a specific date and time
- **Non-interactive background runs** — queued and scheduled jobs do not pause to ask follow-up questions mid-run
- **Task history** — persistent log of every completed task with results and screenshots
- **Activity logs** — detailed step-by-step logs saved per task for review, including per-step usage and a cumulative `total_usage` block
- **Crash-safe recovery** — queued and scheduled tasks persist across restart, while interrupted in-flight tasks are marked failed instead of replayed automatically
- **Screenshot capture** — final page screenshot saved alongside each result
- **Keep-alive browser session** — browser stays open after each task so you can inspect the page
- **Terminal chat** — lightweight CLI interface for quick one-off tasks

## Architecture In Plain English

If you are not a developer, the easiest way to think about this project is as a small local automation stack with four layers:

1. **You** describe a task in normal language
2. **Gradio** provides the web interface where you submit the task and watch progress
3. **LM Studio** runs the local LLM that decides what the next browser action should be
4. **browser-use + Playwright** open and control the browser to carry out those actions

In other words, this app is not a traditional scraper with hard-coded steps. It is an agent that looks at the current page, decides what to do next, performs the action, then repeats until the task is finished.

### What Each Part Does

- **Gradio** is the front end. It gives you the chat-style UI, settings panel, live step updates, queueing, and scheduling.
- **LM Studio** is the local AI endpoint. It hosts the model on your machine and exposes an OpenAI-compatible API that the app calls.
- **browser-use** is the orchestration layer. It turns the LLM's decisions into browser operations.
- **Playwright** is the browser automation engine. It is the component that actually opens pages, clicks buttons, types into fields, scrolls, and captures screenshots.

### How Playwright Is Used In This App

This project **does use Playwright locally**, but **your app code does not call Playwright directly**.

Instead, your Python code calls the `browser-use` library, and `browser-use` uses Playwright internally on the same machine.

That means:

- Playwright is **part of the app runtime**, not a separate external service
- The browser actions happen **locally on your computer**
- You are **not** sending browser control to a cloud API
- LM Studio and Playwright play different roles: LM Studio decides, Playwright executes

From an infrastructure point of view, LM Studio behaves like a local inference endpoint, while Playwright behaves like a local execution engine.

### End-To-End Flow

When you submit a task, the process is roughly:

1. You enter a request such as "find the cheapest flight" or "summarize the top five posts"
2. The app sends that request to the local LLM in LM Studio
3. The LLM decides the next browser action
4. browser-use translates that decision into a concrete browser step
5. Playwright performs the step in the browser
6. The app observes the updated page and repeats the cycle until the task is complete
7. The result, screenshot, open tabs, and activity log are saved for review

### What This App Is And Is Not

- It **is** a local AI-driven browser operator
- It **is** designed for interactive verification, because the browser can stay open after the task finishes
- It **is** suitable for unattended queued and scheduled runs because the app does not stop to ask clarification questions during execution
- It **is not** a fixed workflow automation with predefined selectors for one website only
- It **is not** calling a separate hosted Playwright service
- It **is not** guaranteed to be perfectly reliable on every website, because modern sites change often and can present popups, consent banners, and login challenges

## Recent Changes

- Removed the unfinished preflight clarification path so task execution has one direct flow and background runs remain non-interactive.
- Cleaned up site-specific prompt wording so the agent guidance is more generic across booking, ecommerce, and research sites.
- Changed restart behavior so tasks left in `running` state after a crash are marked failed instead of being re-run automatically on the next launch.
- Extended the activity logger so each step keeps its own usage details and the final JSON now includes a top-level `total_usage` aggregate block.

## Prerequisites

- **Python 3.11+**
- **LM Studio** running with a model loaded and local server started
- A model with good **function/tool calling** support (Qwen 3, OpenAI OSS, etc.)

## Quick Start

### 0. Clone and enter the project

```bash
git clone https://github.com/terenceluk/browser-agent-lmstudio.git
cd Control-Browser-with-Local-LLM
```

### 1. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 3. Configure

Edit `config.py` to match your LM Studio setup:

```python
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
MODEL_NAME = "openai/gpt-oss-120b"  # Must match your loaded model name in LM Studio
```

Current model examples used in this repo:

- `qwen/qwen3.6-35b-a3b`
- `openai/gpt-oss-120b`
- `qwen3-coder-next` (code-focused; usually weaker for browsing tasks)

### 4. Start LM Studio

- Open LM Studio
- Load your model (Qwen 3, OpenAI OSS, etc.)
- Start the local server (default: port 1234)

### 5. Run the app

**Web UI (Gradio):**
```bash
python app.py
```
Then open http://127.0.0.1:7860 in your browser.

**Terminal chat:**
```bash
python chat.py
```

## Usage Examples

```
Go to aircanada.com and find the cheapest flights from Toronto to Vancouver between May 15-30, 2026

Open google.com and search for "best restaurants in Toronto" and list the top 5

Navigate to weather.com and tell me the 5-day forecast for New York City

Go to amazon.ca and find the best-rated wireless headphones under $100

Go to reddit.com/r/LocalLLaMA and summarize the top 5 posts from today
```

## How It Works

1. You type a task in natural language
2. The LLM (running locally in LM Studio) plans a sequence of browser actions
3. Playwright opens Chrome and executes the actions (click, type, scroll, navigate)
4. The agent loops: observe page → plan next action → execute → repeat
5. When done, it returns the result and saves a screenshot

## Configuration Reference

Key settings in `config.py`:

- `LM_STUDIO_BASE_URL`: LM Studio OpenAI-compatible endpoint (default `http://localhost:1234/v1`)
- `MODEL_NAME`: model identifier loaded in LM Studio
- `BROWSER_HEADLESS`: run browser hidden (`True`) or visible (`False`)
- `MAX_STEPS`: max allowed action steps per task
- `USE_VISION`: enable screenshot-aware reasoning for models that support vision
- `BROWSER_WIDTH` / `BROWSER_HEIGHT`: browser window size used by Playwright
- `SHOW_LLM_CONSOLE_TRACE`: prints raw model trace and reasoning content to terminal

## Project Structure

```
Control-Browser-with-Local-LLM/
├── app.py                  # Gradio web UI (full-featured)
├── chat.py                 # Terminal chat interface (lightweight)
├── agent.py                # Browser agent (browser-use wrapper)
├── config.py               # LM Studio & browser settings
├── requirements.txt        # Python dependencies
├── task_history.json       # Auto-generated: history of completed tasks
├── scheduled_tasks.json    # Auto-generated: scheduled task store
├── activity_logs/          # Auto-generated: per-task step logs
└── exports/                # Auto-generated: saved results and screenshots
```

> **Note:** `task_history.json`, `scheduled_tasks.json`, `activity_logs/`, and `exports/` are generated at runtime and are excluded from version control via `.gitignore`.

Activity log files now include:

- Per-step execution traces
- Per-step usage in both string and structured form
- A top-level `total_usage` object with aggregate prompt, completion, reasoning, and total token counts

## Troubleshooting

- **Cannot connect to LM Studio**: verify LM Studio server is running and `LM_STUDIO_BASE_URL` in `config.py` matches the port.
- **Model not found**: set `MODEL_NAME` to the exact identifier shown in LM Studio server tab.
- **Playwright launch issues**: rerun `playwright install chromium` in the active virtual environment.
- **Task stops too early**: increase `MAX_STEPS` in `config.py`.
- **Dynamic sites fail intermittently**: run with `BROWSER_HEADLESS = False` and inspect the live browser plus `activity_logs/` traces.

## Security and Data Handling

- Do not commit real credentials in prompts, logs, screenshots, or docs.
- `activity_logs/`, `exports/`, `task_history.json`, and `scheduled_tasks.json` are runtime artifacts and are ignored by `.gitignore`.
- If you share logs publicly, review and redact personal data first.

## Tips

- **Watch the browser**: Set `BROWSER_HEADLESS = False` in `config.py` to see the agent work in real time
- **Model matters**: Larger models (30B+) work much better for complex multi-step tasks
- **Vision mode**: If your model supports images, set `USE_VISION = True` for screenshot-based navigation
- **Max steps**: Increase `MAX_STEPS` in `config.py` for tasks that need more actions
- **Queue tasks**: Submit multiple tasks from the web UI and let the agent work through them unattended
- **Check logs**: After a task completes, open the saved activity log for a full step-by-step trace of what the agent did
