"""Gradio-based Chat UI for the LM Studio Browser Pilot."""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio as gr
from PIL import Image as PILImage

from agent import BrowserAgentSession, create_lm_studio_llm
from browser_use.llm.models import ChatOpenAI
import config as cfg


ROOT_DIR = Path(__file__).resolve().parent
HISTORY_PATH = ROOT_DIR / "task_history.json"
EXPORTS_DIR = ROOT_DIR / "exports"
ACTIVITY_LOGS_DIR = ROOT_DIR / "activity_logs"
SCHEDULED_TASKS_PATH = ROOT_DIR / "scheduled_tasks.json"
MAX_STEPS = cfg.MAX_STEPS
MAX_HISTORY_ENTRIES = 200
MAX_LOG_FILES = 50
MAX_QUEUE_SLOTS = 4
QUEUE_POLL_SECONDS = 0.4
SCHEDULER_POLL_SECONDS = 1.0

agent_session: BrowserAgentSession | None = None
agent_session_config: tuple[str, str, int] | None = None
scheduler_worker: asyncio.Task | None = None
task_queue: list["QueuedTask"] = []
active_task_info: dict[str, Any] | None = None
restored_persisted_tasks = False
queue_lock = asyncio.Lock()
runner_lock = asyncio.Lock()
_pending_chat_messages: list[dict[str, Any]] = []  # background task results to append to chatbot


@dataclass
class QueuedTask:
    id: str
    task: str
    base_url: str
    model: str
    max_steps: int
    created_at: datetime
    scheduled_for: datetime | None = None
    update_queue: asyncio.Queue | None = None
    result_future: asyncio.Future | None = None
    source: str = "interactive"
    step_messages: list[str] = field(default_factory=list)
    step_traces: list[dict[str, Any]] = field(default_factory=list)
    latest_tabs: list[dict] = field(default_factory=list)


def _read_scheduled_store() -> list[dict]:
    if not SCHEDULED_TASKS_PATH.exists():
        return []

    try:
        payload = json.loads(SCHEDULED_TASKS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _write_scheduled_store(records: list[dict]) -> None:
    SCHEDULED_TASKS_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _iso_or_blank(value: datetime | None) -> str:
    return value.isoformat(timespec="seconds") if value else ""


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _upsert_scheduled_record(record: dict) -> None:
    records = _read_scheduled_store()
    for index, existing in enumerate(records):
        if existing.get("id") == record.get("id"):
            records[index] = record
            break
    else:
        records.append(record)
    _write_scheduled_store(records)


def _get_scheduled_record(task_id: str) -> dict | None:
    for record in _read_scheduled_store():
        if record.get("id") == task_id:
            return record
    return None


def _build_task_record(
    item: QueuedTask,
    *,
    status: str,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    result_text: str = "",
    error_text: str = "",
    log_path: str = "",
    screenshot_path: str = "",
    tabs: list[dict] | None = None,
) -> dict:
    existing = _get_scheduled_record(item.id) or {}
    return {
        "id": item.id,
        "task": item.task,
        "base_url": item.base_url,
        "model": item.model,
        "max_steps": item.max_steps,
        "created_at": existing.get("created_at") or _iso_or_blank(item.created_at),
        "scheduled_for": _iso_or_blank(item.scheduled_for),
        "source": item.source,
        "status": status,
        "started_at": _iso_or_blank(started_at) or existing.get("started_at", ""),
        "finished_at": _iso_or_blank(finished_at) or existing.get("finished_at", ""),
        "result_text": result_text or existing.get("result_text", ""),
        "error_text": error_text or existing.get("error_text", ""),
        "log_path": log_path or existing.get("log_path", ""),
        "screenshot_path": screenshot_path or existing.get("screenshot_path", ""),
        "tabs": tabs if tabs is not None else existing.get("tabs", []),
    }


def _queue_item_label(item: QueuedTask) -> str:
    when = ""
    if item.scheduled_for and item.scheduled_for > datetime.now():
        when = f"[{item.scheduled_for.strftime('%m-%d %H:%M')}] "
    elif item.source == "scheduled":
        when = "[queued] "
    return f"{when}{_short_task_text(item.task, 58)}"


def _queued_task_choices() -> list[tuple[str, str]]:
    items = sorted(task_queue, key=lambda item: (item.scheduled_for or item.created_at, item.created_at))
    choices = [(_queue_item_label(item), item.id) for item in items]
    if not choices:
        return [("No queued tasks", "")]
    return choices


def _queued_task_dropdown_update() -> gr.update:
    choices = _queued_task_choices()
    first_value = choices[0][1] if choices and choices[0][1] else None
    return gr.update(choices=choices, value=first_value, interactive=bool(first_value))


def _retarget_pending_tasks(base_url: str, model: str, max_steps: int) -> None:
    normalized_max_steps = _normalize_max_steps(max_steps)

    for item in task_queue:
        item.base_url = base_url
        item.model = model
        item.max_steps = normalized_max_steps

    records = _read_scheduled_store()
    changed = False
    for record in records:
        if record.get("status") not in {"queued", "scheduled", "running"}:
            continue
        if (
            record.get("base_url") == base_url
            and record.get("model") == model
            and _normalize_max_steps(record.get("max_steps", normalized_max_steps)) == normalized_max_steps
        ):
            continue

        record["base_url"] = base_url
        record["model"] = model
        record["max_steps"] = normalized_max_steps
        changed = True

    if changed:
        _write_scheduled_store(records)


def _scheduled_results_records(limit: int = 5) -> list[dict]:
    records = [
        record for record in _read_scheduled_store()
        if record.get("source") in {"scheduled", "queued"} and record.get("status") in {"completed", "failed", "canceled"}
    ]
    records.sort(
        key=lambda item: item.get("finished_at") or item.get("created_at") or "",
        reverse=True,
    )
    return records[:limit]


def _scheduled_results_summary_md() -> str:
    records = _scheduled_results_records()
    if not records:
        return "No scheduled results yet."

    lines: list[str] = []
    for record in records:
        finished = record.get("finished_at") or record.get("created_at") or ""
        lines.append(
            f"- **{record.get('status', '').upper()}** {finished}: {_short_task_text(record.get('task', ''), 88)}"
        )
    return "\n".join(lines)


def _scheduled_result_choices() -> list[tuple[str, str]]:
    choices = [
        (
            f"{record.get('status', '').upper()} | {_short_task_text(record.get('task', ''), 50)}",
            record.get("id", ""),
        )
        for record in _scheduled_results_records()
    ]
    if not choices:
        return [("No scheduled results", "")]
    return choices


def _scheduled_results_dropdown_update() -> gr.update:
    choices = _scheduled_result_choices()
    first_value = choices[0][1] if choices and choices[0][1] else None
    return gr.update(choices=choices, value=first_value, interactive=bool(first_value))


def _scheduled_result_view(task_id: str) -> str:
    if not task_id:
        return "No scheduled result selected."

    record = _get_scheduled_record(task_id)
    if not record:
        return "Scheduled result not found."

    lines = [
        f"**Status:** {record.get('status', '').upper()}",
        f"**Task:** {record.get('task', '')}",
        f"**Finished:** {record.get('finished_at', '')}",
        f"**Model:** {record.get('model', '')}",
    ]
    if record.get("result_text"):
        lines += ["", "**Result:**", record["result_text"]]
    if record.get("error_text"):
        lines += ["", "**Error:**", record["error_text"]]
    if record.get("screenshot_path"):
        lines += ["", f"**Screenshot:** `{record['screenshot_path']}`"]
    if record.get("log_path"):
        lines += ["", f"**Log File:** `{record['log_path']}`"]
    return "\n".join(lines)


def _scheduled_panel_updates() -> tuple[str, gr.update]:
    return _scheduled_results_summary_md(), _scheduled_results_dropdown_update()


async def _restore_persisted_queue() -> None:
    global restored_persisted_tasks
    if restored_persisted_tasks:
        return

    restored_persisted_tasks = True
    existing_ids = {item.id for item in task_queue}
    for record in _read_scheduled_store():
        status = record.get("status")
        if status == "running":
            # Task was interrupted by a crash or restart — mark it as failed and skip
            record["status"] = "failed"
            record["error_text"] = "Interrupted: the application was restarted before this task could complete."
            record["finished_at"] = datetime.now().isoformat(timespec="seconds")
            _upsert_scheduled_record(record)
            continue
        if status not in {"queued", "scheduled"}:
            continue
        if record.get("id") in existing_ids:
            continue

        scheduled_for = _parse_dt(record.get("scheduled_for"))
        item = QueuedTask(
            id=record.get("id", uuid4().hex),
            task=record.get("task", ""),
            base_url=record.get("base_url", cfg.LM_STUDIO_BASE_URL),
            model=record.get("model", cfg.MODEL_NAME),
            max_steps=_normalize_max_steps(record.get("max_steps", MAX_STEPS)),
            created_at=_parse_dt(record.get("created_at")) or datetime.now(),
            scheduled_for=scheduled_for,
            source=record.get("source", "scheduled"),
        )
        task_queue.append(item)


def _restore_persisted_queue_sync() -> None:
    global restored_persisted_tasks
    if restored_persisted_tasks:
        return

    restored_persisted_tasks = True
    existing_ids = {item.id for item in task_queue}
    for record in _read_scheduled_store():
        status = record.get("status")
        if status == "running":
            # Task was interrupted by a crash or restart — mark it as failed and skip
            record["status"] = "failed"
            record["error_text"] = "Interrupted: the application was restarted before this task could complete."
            record["finished_at"] = datetime.now().isoformat(timespec="seconds")
            _upsert_scheduled_record(record)
            continue
        if status not in {"queued", "scheduled"}:
            continue
        if record.get("id") in existing_ids:
            continue

        item = QueuedTask(
            id=record.get("id", uuid4().hex),
            task=record.get("task", ""),
            base_url=record.get("base_url", cfg.LM_STUDIO_BASE_URL),
            model=record.get("model", cfg.MODEL_NAME),
            max_steps=_normalize_max_steps(record.get("max_steps", MAX_STEPS)),
            created_at=_parse_dt(record.get("created_at")) or datetime.now(),
            scheduled_for=_parse_dt(record.get("scheduled_for")),
            source=record.get("source", "scheduled"),
        )
        task_queue.append(item)


async def _remove_queued_task(task_id: str) -> bool:
    async with queue_lock:
        for index, item in enumerate(task_queue):
            if item.id != task_id:
                continue
            removed = task_queue.pop(index)
            _upsert_scheduled_record(
                _build_task_record(
                    removed,
                    status="canceled",
                    finished_at=datetime.now(),
                    error_text="Canceled by user before execution.",
                )
            )
            return True
    return False


def _normalize_max_steps(max_steps: int | float | None) -> int:
    try:
        value = int(max_steps or MAX_STEPS)
    except (TypeError, ValueError):
        value = MAX_STEPS
    return max(1, min(value, 200))


def _short_task_text(task: str, limit: int = 72) -> str:
    compact = " ".join(task.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _parse_schedule_input(schedule_text: str) -> datetime | None:
    value = schedule_text.strip()
    if not value:
        return None

    for parser in (
        lambda text: datetime.fromisoformat(text),
        lambda text: datetime.strptime(text, "%Y-%m-%d %H:%M"),
        lambda text: datetime.strptime(text, "%Y-%m-%d %H:%M:%S"),
    ):
        try:
            return parser(value)
        except ValueError:
            continue

    raise ValueError("Use YYYY-MM-DD HH:MM or YYYY-MM-DD HH:MM:SS")


def _schedule_date_choices() -> list[tuple[str, str]]:
    """Next 14 days as (label, ISO-date-string) pairs."""
    from datetime import date as _date, timedelta
    today = _date.today()
    choices = []
    for i in range(14):
        d = today + timedelta(days=i)
        if i == 0:
            label = f"Today  ({d.strftime('%b %d, %Y')})"
        elif i == 1:
            label = f"Tomorrow  ({d.strftime('%b %d, %Y')})"
        else:
            label = d.strftime("%A, %b %d, %Y")
        choices.append((label, d.isoformat()))
    return choices


def _schedule_hr_choices() -> list[str]:
    return [str(h) for h in range(1, 13)]


def _schedule_min_choices() -> list[str]:
    return [f"{m:02d}" for m in range(0, 60)]


MONTH_NAME_PATTERN = (
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
BOOKING_KEYWORDS = (
    "book",
    "booking",
    "reserve",
    "reservation",
    "schedule",
    "appointment",
    "slot",
)
def _task_mentions_booking(task: str) -> bool:
    lowered = task.lower()
    return any(keyword in lowered for keyword in BOOKING_KEYWORDS)


def _model_status(base_url: str, model: str) -> tuple[bool, bool, str]:
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=2.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return False, False, f"Offline ({exc.reason})"
    except Exception as exc:
        return False, False, f"Offline ({exc})"

    models = [item.get("id", "") for item in payload.get("data", []) if isinstance(item, dict)]
    model_found = model in models if models else False
    if not models:
        detail = "Online, but LM Studio returned no models"
    elif model_found:
        detail = f"Online, model loaded ({len(models)} model(s) available)"
    else:
        detail = f"Online, but current model not listed ({len(models)} model(s) available)"
    return True, model_found, detail


def _build_llm_client(base_url: str, model: str) -> ChatOpenAI:
    return create_lm_studio_llm(model_name=model, base_url=base_url)


def build_session(base_url: str, model: str, max_steps: int) -> BrowserAgentSession:
    llm = _build_llm_client(base_url, model)
    return BrowserAgentSession(llm=llm, max_steps=_normalize_max_steps(max_steps))


def get_session(base_url: str, model: str, max_steps: int) -> BrowserAgentSession:
    global agent_session, agent_session_config
    config_key = (base_url, model, _normalize_max_steps(max_steps))
    if agent_session is None or agent_session_config != config_key:
        agent_session = build_session(*config_key)
        agent_session_config = config_key
    return agent_session


def _status_md(base_url: str, model: str, max_steps: int) -> str:
    is_online, model_found, detail = _model_status(base_url, model)
    server_status = "Online" if is_online else "Offline"
    model_status = "Found" if model_found else "Missing"
    return (
        f"**URL:** `{base_url}`  \n"
        f"**Model:** `{model}` ({model_status})  \n"
        f"**Max Steps:** `{_normalize_max_steps(max_steps)}`  \n"
        f"**Server:** `{server_status}`  \n"
        f"**Status:** {detail}"
    )


def _steps_md(step_messages: list[str] | None = None) -> str:
    if not step_messages:
        return "No active task."
    return "\n".join(f"- {message}" for message in step_messages)


def _normalize_step_trace(step: Any, fallback_index: int | None = None) -> dict[str, Any]:
    if isinstance(step, str):
        return {
            "index": fallback_index,
            "message": step,
            "content": "",
            "reasoning": "",
            "usage": "",
        }

    if isinstance(step, dict):
        message = str(step.get("message", "") or "").strip()
        return {
            "index": step.get("index", fallback_index),
            "message": message,
            "content": str(step.get("content", "") or ""),
            "reasoning": str(step.get("reasoning", "") or ""),
            "usage": str(step.get("usage", "") or ""),
            "model": str(step.get("model", "") or ""),
            "structured": bool(step.get("structured", False)),
        }

    return {
        "index": fallback_index,
        "message": str(step or "").strip(),
        "content": "",
        "reasoning": "",
        "usage": "",
    }


def _tabs_md(tabs: list[dict] | None = None) -> str:
    if not tabs:
        return "No tabs captured yet."

    lines = []
    for index, tab in enumerate(tabs, start=1):
        title = (tab.get("title") or "Untitled").replace("\n", " ").strip()
        url = (tab.get("url") or "").strip()
        if url:
            lines.append(f"{index}. **{title}**  \n   <{url}>")
        else:
            lines.append(f"{index}. **{title}**")
    return "\n".join(lines)


def load_task_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []

    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _history_summary_md() -> str:
    history = load_task_history()
    if not history:
        return "No saved tasks yet."

    recent = history[-5:][::-1]
    lines = []
    for item in recent:
        task = item.get("task", "").strip().replace("\n", " ")
        if len(task) > 90:
            task = task[:90] + "..."
        lines.append(f"- **{item.get('timestamp', '')}**: {task}")
    return "\n".join(lines)


def _history_button_items(limit: int = 5) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for entry in load_task_history()[-limit:][::-1]:
        task = entry.get("task", "").strip()
        items.append((_short_task_text(task, limit=44), task))

    while len(items) < limit:
        items.append(("No saved task", ""))
    return items


def _history_control_updates() -> list[Any]:
    updates: list[Any] = [_history_summary_md()]
    for label, task in _history_button_items():
        updates.append(gr.update(value=label, interactive=bool(task)))
    for _, task in _history_button_items():
        updates.append(task)
    return updates


def _queue_summary_md() -> str:
    lines: list[str] = []

    if active_task_info:
        lines.append(
            f"**Running now:** {_short_task_text(active_task_info['task'])}"
            f"  \nStarted: {active_task_info['started_at']}"
        )
    else:
        lines.append("**Running now:** None")

    scheduled = sorted(
        task_queue,
        key=lambda item: (item.scheduled_for or item.created_at, item.created_at),
    )
    if not scheduled:
        lines.append("\n**Queued / Scheduled:** None")
        return "\n".join(lines)

    lines.append("\n**Queued / Scheduled:**")
    now = datetime.now()
    for index, item in enumerate(scheduled, start=1):
        if item.scheduled_for and item.scheduled_for > now:
            when = item.scheduled_for.strftime("%Y-%m-%d %H:%M")
            prefix = f"{index}. Scheduled for {when}"
        else:
            prefix = f"{index}. Waiting"
        lines.append(f"- {prefix}: {_short_task_text(item.task)}")
    return "\n".join(lines)


def save_task_history_entry(task: str, result_text: str, tabs: list[dict]) -> None:
    history = load_task_history()
    history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "task": task,
            "result": result_text,
            "tabs": tabs,
        }
    )
    history = history[-MAX_HISTORY_ENTRIES:]
    HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_activity_screenshot(started_at: datetime, screenshot_bytes: bytes) -> Path:
    ACTIVITY_LOGS_DIR.mkdir(exist_ok=True)
    screenshot_path = ACTIVITY_LOGS_DIR / f"activity_{started_at.strftime('%Y%m%d_%H%M%S')}.png"
    screenshot_path.write_bytes(screenshot_bytes)
    return screenshot_path


def _prune_activity_logs() -> None:
    ACTIVITY_LOGS_DIR.mkdir(exist_ok=True)
    log_files = sorted(ACTIVITY_LOGS_DIR.glob("activity_*.json"))
    if len(log_files) <= MAX_LOG_FILES:
        return

    for log_path in log_files[:-MAX_LOG_FILES]:
        png_path = log_path.with_suffix(".png")
        if png_path.exists():
            png_path.unlink(missing_ok=True)
        log_path.unlink(missing_ok=True)


def save_activity_log(
    *,
    started_at: datetime,
    finished_at: datetime,
    task: str,
    base_url: str,
    model: str,
    step_messages: list[str],
    step_traces: list[dict[str, Any]] | None = None,
    status: str,
    max_steps: int,
    result_text: str = "",
    tabs: list[dict] | None = None,
    screenshot_path: Path | None = None,
    error_text: str | None = None,
    console_trace: dict[str, Any] | None = None,
) -> Path:
    ACTIVITY_LOGS_DIR.mkdir(exist_ok=True)
    log_path = ACTIVITY_LOGS_DIR / f"activity_{started_at.strftime('%Y%m%d_%H%M%S')}.json"

    def parse_usage(usage_val):
        """Parse usage string to dict. Accepts string or dict. Returns dict with all keys as int, missing as 0."""
        if isinstance(usage_val, dict):
            # Already structured
            return {
                "prompt_tokens": int(usage_val.get("prompt_tokens", 0)),
                "completion_tokens": int(usage_val.get("completion_tokens", 0)),
                "reasoning_tokens": int(usage_val.get("reasoning_tokens", 0)),
                "total_tokens": int(usage_val.get("total_tokens", 0)),
            }
        if not isinstance(usage_val, str):
            return {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
        # Parse string like "prompt_tokens: 12739\ncompletion_tokens: 607\n..."
        result = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
        for line in usage_val.split("\n"):
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k in result:
                try:
                    result[k] = int(v)
                except Exception:
                    pass
        return result

    # Normalize steps and parse usage
    norm_steps = []
    for index, step in enumerate(step_traces or step_messages, start=1):
        norm = _normalize_step_trace(step, fallback_index=index)
        usage_val = norm.get("usage", None)
        usage_struct = parse_usage(usage_val)
        # Store both for compatibility
        norm["usage_struct"] = usage_struct
        norm["usage_str"] = usage_val if isinstance(usage_val, str) else None
        norm_steps.append(norm)

    # Compute totals
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0, "step_count": len(norm_steps)}
    for step in norm_steps:
        u = step.get("usage_struct", {})
        for k in ["prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"]:
            total_usage[k] += int(u.get(k, 0))

    payload = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "status": status,
        "task": task,
        "settings": {
            "base_url": base_url,
            "model": model,
            "max_steps": _normalize_max_steps(max_steps),
        },
        "steps": norm_steps,
        "total_usage": total_usage,
        "result": result_text,
        "tabs": tabs or [],
        "screenshot_path": str(screenshot_path) if screenshot_path else "",
        "error": error_text,
        "console_trace": console_trace or {},
    }
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _prune_activity_logs()
    return log_path


def load_latest_log() -> str:
    """Return the contents of the most recent activity log, formatted for display."""
    if not ACTIVITY_LOGS_DIR.exists():
        return "No activity logs found yet."

    logs = sorted(ACTIVITY_LOGS_DIR.glob("activity_*.json"))
    if not logs:
        return "No activity logs found yet."

    try:
        data = json.loads(logs[-1].read_text(encoding="utf-8"))
    except Exception as exc:
        return f"Could not read log: {exc}"

    lines = [
        f"**File:** `{logs[-1].name}`",
        f"**Task:** {data.get('task', '').strip()}",
        f"**Status:** {data.get('status', '').upper()}",
        f"**Started:** {data.get('started_at', '')}",
        f"**Duration:** {data.get('duration_seconds', '')}s",
        f"**Model:** {data.get('settings', {}).get('model', '')}",
        f"**Max Steps:** {data.get('settings', {}).get('max_steps', '')}",
        "",
        "**Steps:**",
    ]
    for step in data.get("steps", []):
        lines.append(f"  {step['index']}. {step['message']}")
        if step.get("content"):
            lines.append(f"     Content: {step['content']}")
        if step.get("reasoning"):
            lines.append(f"     Reasoning: {step['reasoning']}")
        if step.get("usage"):
            lines.append(f"     Usage: {step['usage']}")

    if data.get("result"):
        lines += ["", "**Result:**", data["result"]]
    if data.get("error"):
        lines += ["", "**Error:**", data["error"]]
    if data.get("screenshot_path"):
        lines += ["", f"**Screenshot file:** `{data['screenshot_path']}`"]
    if data.get("tabs"):
        lines += ["", "**Tabs at completion:"]
        for tab in data["tabs"]:
            lines.append(f"  - {tab.get('title', 'Untitled')} - {tab.get('url', '')}")

    return "\n".join(lines)


def export_last_result(task: str, result_text: str, tabs: list[dict]) -> gr.update:
    if not result_text.strip():
        return gr.update(value=None, visible=False)

    EXPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = EXPORTS_DIR / f"browser_agent_result_{timestamp}.txt"

    lines = [
        f"Task: {task}",
        f"Exported: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Result:",
        result_text,
    ]
    if tabs:
        lines.extend(["", "Tabs:"])
        for index, tab in enumerate(tabs, start=1):
            lines.append(f"{index}. {tab.get('title', 'Untitled')} - {tab.get('url', '')}")

    export_path.write_text("\n".join(lines), encoding="utf-8")
    return gr.update(value=str(export_path), visible=True)


async def _ensure_scheduler_started() -> None:
    global scheduler_worker
    await _restore_persisted_queue()
    if scheduler_worker is None or scheduler_worker.done():
        scheduler_worker = asyncio.create_task(_scheduler_loop())


async def _enqueue_task(item: QueuedTask) -> None:
    async with queue_lock:
        task_queue.append(item)
    initial_status = "scheduled" if item.scheduled_for and item.scheduled_for > datetime.now() else "queued"
    _upsert_scheduled_record(_build_task_record(item, status=initial_status))


async def _pop_next_ready_task() -> QueuedTask | None:
    now = datetime.now()
    async with queue_lock:
        ready_items = [
            item for item in task_queue
            if item.scheduled_for is None or item.scheduled_for <= now
        ]
        if not ready_items:
            return None

        ready_items.sort(key=lambda item: (item.scheduled_for or item.created_at, item.created_at))
        next_item = ready_items[0]
        task_queue.remove(next_item)
        _upsert_scheduled_record(
            _build_task_record(next_item, status="running", started_at=datetime.now())
        )
        return next_item


async def _queue_position(task_id: str) -> int | None:
    async with queue_lock:
        for index, item in enumerate(task_queue, start=1):
            if item.id == task_id:
                return index
    return None


async def _run_queued_task(item: QueuedTask) -> dict[str, Any]:
    global active_task_info

    started_at = datetime.now()
    active_task_info = {
        "id": item.id,
        "task": item.task,
        "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
    }

    session = get_session(item.base_url, item.model, item.max_steps)
    if item.update_queue is None:
        _pending_chat_messages.append({
            "role": "assistant",
            "content": f"Working on **{_short_task_text(item.task, 120)}**...",
        })
    if item.update_queue is not None:
        await item.update_queue.put({"type": "started"})

    async def on_step(step_update: Any):
        step_trace = _normalize_step_trace(step_update, fallback_index=len(item.step_messages) + 1)
        message_text = step_trace["message"]
        item.step_messages.append(message_text)
        item.step_traces.append(step_trace)
        latest_tabs = await session.get_tabs()
        item.latest_tabs = latest_tabs
        if item.update_queue is not None:
            await item.update_queue.put(
                {
                    "type": "step",
                    "message": message_text,
                    "step": step_trace,
                    "count": len(item.step_messages),
                    "tabs": latest_tabs,
                }
            )

    try:
        result = await session.run_task(task=item.task, step_callback=on_step)
        finished_at = datetime.now()
        screenshot_path = None
        if result.screenshot:
            screenshot_path = _save_activity_screenshot(started_at, result.screenshot)

        save_task_history_entry(item.task, result.text, result.tabs)
        log_path = save_activity_log(
            started_at=started_at,
            finished_at=finished_at,
            task=item.task,
            base_url=item.base_url,
            model=item.model,
            max_steps=item.max_steps,
            step_messages=item.step_messages,
            step_traces=item.step_traces,
            status="completed",
            result_text=result.text,
            tabs=result.tabs,
            screenshot_path=screenshot_path,
            console_trace=result.console_trace,
        )
        payload = {
            "status": "completed",
            "task": item.task,
            "task_id": item.id,
            "result_text": result.text,
            "tabs": result.tabs,
            "screenshot": result.screenshot,
            "log_path": str(log_path),
            "history_md": _history_summary_md(),
            "queue_md": _queue_summary_md(),
            "history_controls": _history_control_updates(),
            "scheduled_panel": _scheduled_panel_updates(),
            "screenshot_path": str(screenshot_path) if screenshot_path else "",
            "steps_md": _steps_md(item.step_messages),
            "step_count": len(item.step_messages),
        }
        _upsert_scheduled_record(
            _build_task_record(
                item,
                status="completed",
                started_at=started_at,
                finished_at=finished_at,
                result_text=result.text,
                log_path=str(log_path),
                screenshot_path=str(screenshot_path) if screenshot_path else "",
                tabs=result.tabs,
            )
        )
        if item.update_queue is not None:
            await item.update_queue.put({"type": "completed", "payload": payload})
        else:
            _pending_chat_messages.append({
                "role": "assistant",
                "content": (
                    f"**{_short_task_text(item.task, 120)}**\n\n"
                    f"**Result:**\n\n{result.text}"
                ),
            })
        return payload
    except Exception as exc:
        finished_at = datetime.now()
        log_path = save_activity_log(
            started_at=started_at,
            finished_at=finished_at,
            task=item.task,
            base_url=item.base_url,
            model=item.model,
            max_steps=item.max_steps,
            step_messages=item.step_messages,
            step_traces=item.step_traces,
            status="failed",
            error_text=str(exc),
            console_trace=session._get_last_llm_trace(),
        )
        payload = {
            "status": "failed",
            "task": item.task,
            "task_id": item.id,
            "error_text": str(exc),
            "tabs": item.latest_tabs,
            "screenshot": None,
            "log_path": str(log_path),
            "history_md": _history_summary_md(),
            "queue_md": _queue_summary_md(),
            "history_controls": _history_control_updates(),
            "scheduled_panel": _scheduled_panel_updates(),
            "steps_md": _steps_md(item.step_messages),
            "step_count": len(item.step_messages),
        }
        _upsert_scheduled_record(
            _build_task_record(
                item,
                status="failed",
                started_at=started_at,
                finished_at=finished_at,
                error_text=str(exc),
                log_path=str(log_path),
                tabs=item.latest_tabs,
            )
        )
        if item.update_queue is not None:
            await item.update_queue.put({"type": "completed", "payload": payload})
        else:
            _pending_chat_messages.append({
                "role": "assistant",
                "content": f"**{_short_task_text(item.task, 120)}**\n\n\u274c Failed: {str(exc)}",
            })
        return payload
    finally:
        active_task_info = None


async def _scheduler_loop() -> None:
    while True:
        if runner_lock.locked():
            await asyncio.sleep(SCHEDULER_POLL_SECONDS)
            continue

        next_item = await _pop_next_ready_task()
        if next_item is None:
            await asyncio.sleep(SCHEDULER_POLL_SECONDS)
            continue

        async with runner_lock:
            payload = await _run_queued_task(next_item)
            if next_item.result_future is not None and not next_item.result_future.done():
                next_item.result_future.set_result(payload)


def _drain_pending_chat_messages(chat_history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    merged_history = list(chat_history or [])
    while _pending_chat_messages:
        merged_history.append(_pending_chat_messages.pop(0))
    return merged_history


def _queue_slot_md(task_id: str) -> str:
    """Render the content of a queue-slot tab from the persisted store."""
    if not task_id:
        return ""
    record = _get_scheduled_record(task_id)
    if not record:
        return "Loading..."

    task_text = record.get("task", "").strip()
    status = record.get("status", "").upper()

    icon = {"QUEUED": "⏳", "SCHEDULED": "🗓", "RUNNING": "⚙️", "COMPLETED": "✅", "FAILED": "❌", "CANCELED": "🚫"}.get(status, "")
    lines = [
        f"### {task_text}",
        "",
        f"**Status:** {icon} {status}",
    ]
    if record.get("started_at"):
        lines.append(f"**Started:** {record['started_at']}")
    if record.get("finished_at"):
        lines.append(f"**Finished:** {record['finished_at']}")
    if record.get("scheduled_for"):
        lines.append(f"**Scheduled for:** {record['scheduled_for']}")

    if status in ("QUEUED", "SCHEDULED"):
        lines += ["", "*Waiting for previous task to finish. This panel refreshes every 5 seconds.*"]
    elif status == "RUNNING":
        lines += ["", "*Running — check Live Progress in the sidebar. This panel refreshes every 5 seconds.*"]
    elif status == "COMPLETED" and record.get("result_text"):
        lines += ["", "**Result:**", record["result_text"]]
        if record.get("screenshot_path"):
            lines += ["", f"*Screenshot saved to `{record['screenshot_path']}`*"]
    elif status == "FAILED" and record.get("error_text"):
        lines += ["", "**Error:**", record["error_text"]]
    elif status == "CANCELED":
        lines += ["", "*This task was canceled before it ran.*"]

    return "\n".join(lines)


def _queue_slot_label(task_id: str) -> str:
    record = _get_scheduled_record(task_id)
    if not record:
        return "Job"
    return _short_task_text(record.get("task", "Job"), 22)


def _find_available_port(preferred_port: int = 7860, host: str = "127.0.0.1") -> int:
    env_port = os.getenv("GRADIO_SERVER_PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            pass

    for port in range(preferred_port, preferred_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


CSS = """
#theme-toggle-btn {
    background: var(--app-panel);
    border: 1px solid var(--app-border);
    border-radius: 999px;
    color: var(--app-subtext);
    padding: 7px 16px;
    font-size: 0.82rem;
    cursor: pointer;
}

#theme-toggle-btn:hover {
    border-color: var(--app-accent);
    color: var(--app-accent);
}
:root.dark {
    --background-fill-primary: #0d1117;
    --background-fill-secondary: #161b22;
    --body-background-fill: #0d1117;
    --block-background-fill: #161b22;
    --panel-background-fill: #161b22;
    --input-background-fill: #0d1117;
    --input-background-fill-focus: #0d1117;
    --input-background-fill-hover: #0d1117;
    --border-color-primary: #30363d;
    --border-color-accent: #388bfd;
    --body-text-color: #e6edf3;
    --input-border-color: #30363d;
    --input-border-color-focus: #58a6ff;
    --input-placeholder-color: #484f58;
    --block-label-text-color: #8b949e;
    --block-title-text-color: #8b949e;
    --button-secondary-background-fill: #21262d;
    --button-secondary-background-fill-hover: #30363d;
    --button-secondary-border-color: #30363d;
    --button-secondary-text-color: #c9d1d9;
    --color-accent: #388bfd;
    --color-accent-soft: #1f2937;
    --app-bg: #0f1117;
    --app-panel: #161b22;
    --app-border: #30363d;
    --app-subtext: #8b949e;
    --app-accent: #58a6ff;
    --app-accent2: #9b6dff;
    --app-success: #238636;
    --app-danger: #f85149;
    --app-alt: #0d1117;
}

:root:not(.dark) {
    --background-fill-primary: #ffffff;
    --background-fill-secondary: #f6f8fa;
    --body-background-fill: #f6f8fa;
    --block-background-fill: #ffffff;
    --panel-background-fill: #f6f8fa;
    --input-background-fill: #ffffff;
    --input-background-fill-focus: #ffffff;
    --input-background-fill-hover: #ffffff;
    --border-color-primary: #d0d7de;
    --border-color-accent: #0969da;
    --body-text-color: #0d1117;
    --input-border-color: #d0d7de;
    --input-border-color-focus: #0969da;
    --input-placeholder-color: #6e7781;
    --block-label-text-color: #57606a;
    --block-title-text-color: #57606a;
    --button-secondary-background-fill: #f6f8fa;
    --button-secondary-background-fill-hover: #eaeef2;
    --button-secondary-border-color: #d0d7de;
    --button-secondary-text-color: #24292f;
    --color-accent: #0969da;
    --color-accent-soft: #ddf4ff;
    --app-bg: #f6f8fa;
    --app-panel: #ffffff;
    --app-border: #d0d7de;
    --app-subtext: #57606a;
    --app-accent: #0969da;
    --app-accent2: #6639ba;
    --app-success: #1a7f37;
    --app-danger: #cf222e;
    --app-alt: #f6f8fa;
}

body, .gradio-container {
    background: radial-gradient(circle at top left, color-mix(in srgb, var(--app-accent) 10%, transparent), transparent 28%), var(--app-bg) !important;
    font-family: "Segoe UI", system-ui, sans-serif;
}

#header {
    background: color-mix(in srgb, var(--app-panel) 92%, transparent);
    border-bottom: 1px solid var(--app-border);
    padding: 16px 24px;
    backdrop-filter: blur(10px);
}


/* Toggle switch styles */
#theme-switch-label {
    user-select: none;
}
#theme-switch-label input[type="checkbox"]:focus + #theme-switch-slider {
    outline: 2px solid var(--app-accent);
}
#theme-switch-slider {
    cursor: pointer;
    background: var(--app-border);
    transition: background 0.2s;
}
#theme-switch-label input[type="checkbox"]:checked + #theme-switch-slider {
    background: var(--app-accent);
}
#theme-switch-slider span {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 18px;
    height: 18px;
    background: var(--app-accent);
    border-radius: 50%;
    transition: left 0.2s, background 0.2s;
}

#main-row {
    gap: 16px !important;
    padding: 16px !important;
}

#chat-col, #settings-col {
    background: var(--app-panel) !important;
    border: 1px solid var(--app-border) !important;
    border-radius: 16px !important;
    overflow: hidden;
}

#chips-row {
    background: var(--app-panel) !important;
    border-bottom: 1px solid var(--app-border) !important;
    padding: 8px 14px 6px !important;
    gap: 6px !important;
    flex-wrap: wrap !important;
}

#chips-label {
    color: var(--app-subtext);
    font-size: 0.75rem;
    white-space: nowrap;
}

.chip button {
    background: var(--app-alt) !important;
    border: 1px solid var(--app-border) !important;
    border-radius: 999px !important;
    color: var(--app-subtext) !important;
    font-size: 0.74rem !important;
    min-width: unset !important;
    height: auto !important;
}

.chip button:hover {
    border-color: var(--app-accent) !important;
    color: var(--app-accent) !important;
}

#chatbot {
    background: var(--app-alt) !important;
    border: none !important;
}

#input-row {
    background: var(--app-panel) !important;
    border-top: 1px solid var(--app-border) !important;
    padding: 12px 14px !important;
    gap: 8px !important;
}

#send-btn {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

#queue-btn {
    background: linear-gradient(135deg, #6e40c9, #9b6dff) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

#clear-btn, #close-btn {
    background: var(--app-panel) !important;
    border: 1px solid var(--app-border) !important;
    border-radius: 10px !important;
    color: var(--app-subtext) !important;
}

#clear-btn:hover {
    border-color: var(--app-subtext) !important;
    color: var(--body-text-color) !important;
}

#close-btn:hover {
    border-color: var(--app-danger) !important;
    color: var(--app-danger) !important;
}

#settings-col {
    padding: 20px !important;
    min-width: 360px;
    max-width: 430px;
}

.settings-section-title {
    color: var(--app-accent) !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 14px 0 6px 0 !important;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--app-border);
}

.schedule-subtitle {
    color: var(--body-text-color);
    font-size: 0.875rem;
    font-weight: 500;
    line-height: 1.25;
    margin: 0 0 6px 0;
}

#apply-btn {
    background: var(--app-success) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

#schedule-btn {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

.secondary-btn button {
    background: var(--app-panel) !important;
    border: 1px solid var(--app-border) !important;
    border-radius: 10px !important;
    color: var(--app-subtext) !important;
}

.secondary-btn button:hover {
    border-color: var(--app-accent) !important;
    color: var(--app-accent) !important;
}

#status-badge,
#queue-md,
#tabs-md,
#history-md {
    border-radius: 10px;
    border: 1px solid var(--app-border);
    background: var(--app-alt);
    padding: 10px 12px;
}

#scheduled-results-summary,
#scheduled-result-viewer {
    border-radius: 10px;
    border: 1px solid var(--app-border);
    background: var(--app-alt);
    padding: 10px 12px;
}

#scheduled-result-viewer {
    max-height: 220px;
    overflow-y: auto;
}

#live-steps {
    border-radius: 10px;
    border: 1px solid var(--app-border);
    background: var(--app-alt);
    padding: 10px 12px;
    max-height: 260px;
    overflow-y: auto;
}

#step-counter {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 36px;
    height: 36px;
    border-radius: 999px;
    border: 1px solid var(--app-border);
    background: var(--app-alt);
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--app-accent);
}

#log-viewer {
    border-radius: 10px;
    border: 1px solid var(--app-border);
    background: var(--app-alt);
    padding: 10px 12px;
    font-size: 0.78rem;
    max-height: 320px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}

#schedule-help {
    font-size: 0.76rem;
    color: var(--app-subtext);
}

#queue-actions,
#scheduled-result-actions {
    gap: 8px !important;
}

#schedule-picker-group {
    gap: 0 !important;
}

#schedule-date {
    margin-bottom: 4px !important;
}

#schedule-date,
#schedule-date > div,
#schedule-date .wrap {
    box-shadow: none !important;
}

#schedule-date .wrap {
    border-bottom-color: transparent !important;
}

#schedule-time-box {
    border: 1px solid var(--app-border) !important;
    background: var(--app-panel) !important;
    border-radius: 10px !important;
    padding: 8px 10px 10px !important;
    gap: 0 !important;
}

#schedule-time-row {
    gap: 6px !important;
    flex-wrap: nowrap !important;
    margin-top: 0 !important;
}

#schedule-time-row > * {
    min-width: 0 !important;
}

#schedule-hr,
#schedule-min,
#schedule-ampm {
    min-width: 0 !important;
}

#cancel-queue-btn {
    background: var(--app-danger) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

#history-buttons {
    gap: 6px !important;
}

.history-task-btn button {
    width: 100% !important;
    justify-content: flex-start !important;
    background: var(--app-alt) !important;
    border: 1px solid var(--app-border) !important;
    border-radius: 10px !important;
    color: var(--body-text-color) !important;
    font-size: 0.78rem !important;
    min-height: 38px !important;
}

.history-task-btn button:hover {
    border-color: var(--app-accent) !important;
}

#main-col {
#screenshot-panel {
    margin-top: 4px !important;
}

#screenshot-panel img {
    border-radius: 8px;
    width: 100%;
}

footer, .footer, [data-testid="footer"], .built-with, .screen-recorder-btn,
.pwa-install-btn, .share-button-container, .gradio-footer {
    display: none !important;
}
"""


JS = """
() => {
  document.documentElement.classList.add('dark');

  document.addEventListener('keydown', function(event) {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      const sendButton = document.querySelector('#send-btn button, #send-btn');
      if (sendButton) {
        event.preventDefault();
        sendButton.click();
      }
    }
  });

  // Auto-scroll live-steps panel to bottom whenever its content changes,
  // unless the user has manually scrolled up to review earlier steps.
  function setupLiveStepsScroll() {
    const el = document.querySelector('#live-steps');
    if (!el) return;
    let userScrolledUp = false;
    el.addEventListener('scroll', function() {
      userScrolledUp = el.scrollTop + el.clientHeight < el.scrollHeight - 10;
    });
    const observer = new MutationObserver(function() {
      if (!userScrolledUp) {
        el.scrollTop = el.scrollHeight;
      }
    });
    observer.observe(el, { childList: true, subtree: true });
  }
  // Gradio renders components asynchronously, so wait briefly before hooking in.
  setTimeout(setupLiveStepsScroll, 1500);
}
"""


EXAMPLE_TASKS = [
    ("CNN headlines", "Go to cnn.com and get the top 3 headlines"),
    ("Amazon search", "Go to amazon.ca and find the best rated wireless headphones under $100"),
    ("Weather", "Go to weather.com and check the 5-day forecast for Toronto, Ontario"),
    ("Flight prices", "Go to aircanada.com and find the cheapest one-way flights from Toronto to Vancouver this week"),
]


def create_ui():
    _restore_persisted_queue_sync()
    _retarget_pending_tasks(cfg.LM_STUDIO_BASE_URL, cfg.MODEL_NAME, MAX_STEPS)
    history_button_items = _history_button_items()

    with gr.Blocks(title="Browser Pilot - LM Studio", css=CSS, js=JS) as app:
        last_task_state = gr.State("")
        last_result_state = gr.State("")
        last_tabs_state = gr.State([])
        last_log_path_state = gr.State("")
        history_task_states = [gr.State(task) for _, task in history_button_items]

        with gr.Row(elem_id="header"):
                        gr.HTML(
                                """
                                <div style='display:flex;align-items:center;justify-content:space-between;width:100%'>
                                    <div>
                                        <h1 style='font-size:1.6rem;font-weight:800;margin:0 0 4px 0;letter-spacing:-0.02em;
                                                             background:linear-gradient(90deg,var(--app-accent),var(--app-accent2));
                                                             -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'>
                                            Browser Pilot
                                        </h1>
                                        <p style='margin:0;font-size:0.82rem;color:var(--app-subtext);'>
                                            Local browser automation with LM Studio, live step tracking, queueing, scheduling, saved history, and reusable exports.
                                        </p>
                                    </div>
                                    <button id="theme-toggle-btn"
                                        onclick="(function(btn){
                                            var root = document.documentElement;
                                            var dark = root.classList.contains('dark');
                                            var nextDark = !dark;
                                            root.classList.toggle('dark', nextDark);
                                            btn.textContent = nextDark ? 'Dark' : 'Light';
                                        })(this)">Light</button>
                                    <script>
                                        (function(){
                                            var btn = document.getElementById('theme-toggle-btn');
                                            if (!btn) return;
                                            btn.textContent = document.documentElement.classList.contains('dark') ? 'Dark' : 'Light';
                                        })();
                                    </script>
                                </div>
                                """
                        )

        with gr.Row(elem_id="main-row", equal_height=False):
            with gr.Column(scale=3, elem_id="chat-col"):
                with gr.Row(elem_id="chips-row"):
                    gr.HTML("<span id='chips-label'>Quick starts:</span>")
                    chip_buttons = [
                        gr.Button(label, size="sm", elem_classes=["chip"])
                        for label, _ in EXAMPLE_TASKS
                    ]

                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    height=620,
                    show_label=False,
                    placeholder=(
                        "<div style='text-align:center;padding:72px 0;opacity:0.4'>"
                        "<div style='font-size:2.2rem'>Browse</div>"
                        "<div style='font-size:1rem;margin-top:8px'>Describe a task to start</div>"
                        "<div style='font-size:0.82rem;margin-top:6px'>Press Ctrl+Enter to send</div>"
                        "</div>"
                    ),
                )

                with gr.Row(elem_id="input-row"):
                    msg = gr.Textbox(
                        elem_id="msg-box",
                        placeholder="Describe a browser task...",
                        show_label=False,
                        scale=6,
                        lines=1,
                        max_lines=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send", elem_id="send-btn", scale=1)
                    queue_btn = gr.Button("Queue", elem_id="queue-btn", scale=1)
                    clear_btn = gr.Button("Clear", elem_id="clear-btn", scale=1)
                    close_btn = gr.Button("Close Browser", elem_id="close-btn", scale=1)

            with gr.Column(scale=1, elem_id="settings-col", min_width=360):
                gr.HTML("<div class='settings-section-title'>LM Studio</div>")
                base_url_in = gr.Textbox(
                    label="Server URL",
                    value=cfg.LM_STUDIO_BASE_URL,
                    placeholder="http://localhost:1234/v1",
                )
                model_in = gr.Textbox(
                    label="Model Name",
                    value=cfg.MODEL_NAME,
                    placeholder="e.g. qwen/qwen3.5-35b-a3b",
                )
                max_steps_in = gr.Number(
                    label="Max Steps",
                    value=MAX_STEPS,
                    precision=0,
                    minimum=1,
                    maximum=200,
                )
                with gr.Row():
                    apply_btn = gr.Button("Apply Settings", elem_id="apply-btn")
                    refresh_status_btn = gr.Button("Refresh Status", elem_classes=["secondary-btn"])
                status_md = gr.Markdown(
                    value=_status_md(cfg.LM_STUDIO_BASE_URL, cfg.MODEL_NAME, MAX_STEPS),
                    elem_id="status-badge",
                )

                gr.HTML("<div class='settings-section-title'>Queue And Schedule</div>")
                queue_md = gr.Markdown(_queue_summary_md(), elem_id="queue-md")
                queue_task_picker = gr.Dropdown(
                    label="Queued Tasks",
                    choices=_queued_task_choices(),
                    value=None,
                    allow_custom_value=False,
                )
                with gr.Row(elem_id="queue-actions"):
                    cancel_queue_btn = gr.Button("Remove Selected", elem_id="cancel-queue-btn")
                with gr.Column(elem_id="schedule-picker-group"):
                    schedule_date_in = gr.DateTime(
                        label="Date",
                        include_time=False,
                        type="string",
                        elem_id="schedule-date",
                    )
                    with gr.Column(elem_id="schedule-time-box"):
                        gr.HTML("<div class='schedule-subtitle'>Time</div>")
                        with gr.Row(elem_id="schedule-time-row"):
                            schedule_hr_in = gr.Dropdown(
                                show_label=False,
                                choices=_schedule_hr_choices(),
                                value="12",
                                allow_custom_value=False,
                                scale=1,
                                min_width=72,
                                elem_id="schedule-hr",
                            )
                            schedule_min_in = gr.Dropdown(
                                show_label=False,
                                choices=_schedule_min_choices(),
                                value="00",
                                allow_custom_value=False,
                                scale=1,
                                min_width=72,
                                elem_id="schedule-min",
                            )
                            schedule_ampm_in = gr.Dropdown(
                                show_label=False,
                                choices=["AM", "PM"],
                                value="AM",
                                allow_custom_value=False,
                                scale=1,
                                min_width=82,
                                elem_id="schedule-ampm",
                            )
                schedule_btn = gr.Button("Schedule Task", elem_id="schedule-btn")

                gr.HTML("<div class='settings-section-title'>Live Progress</div>")
                step_counter = gr.HTML("<div id='step-counter'>0</div>")
                live_steps_md = gr.Markdown("No active task.", elem_id="live-steps")

                gr.HTML("<div class='settings-section-title'>Open Tabs</div>")
                tabs_md = gr.Markdown("No tabs captured yet.", elem_id="tabs-md")

                gr.HTML("<div class='settings-section-title'>Last Screenshot</div>")
                screenshot_img = gr.Image(
                    elem_id="screenshot-panel",
                    label=None,
                    show_label=False,
                    interactive=False,
                    visible=False,
                    type="pil",
                    buttons=["fullscreen", "download"],
                )
                screenshot_hint = gr.HTML(
                    "<div style='text-align:center;padding:16px;font-size:0.78rem;color:var(--app-subtext);border:1px dashed var(--app-border);border-radius:8px;'>No screenshot yet</div>",
                    visible=True,
                )

                gr.HTML("<div class='settings-section-title'>Task History</div>")
                history_md = gr.Markdown(_history_summary_md(), elem_id="history-md")
                history_buttons: list[gr.Button] = []
                with gr.Column(elem_id="history-buttons"):
                    for label, task in history_button_items:
                        history_buttons.append(
                            gr.Button(
                                label,
                                size="sm",
                                interactive=bool(task),
                                elem_classes=["history-task-btn"],
                            )
                        )

                gr.HTML("<div class='settings-section-title'>Export</div>")
                export_btn = gr.Button("Export Last Result")
                export_file = gr.File(label="Exported file", visible=False)

                gr.HTML("<div class='settings-section-title'>Last Activity Log</div>")
                open_log_btn = gr.Button("Load Latest Log")
                log_viewer = gr.Markdown(
                    "Click \"Load Latest Log\" to inspect the most recent run.",
                    elem_id="log-viewer",
                )

                gr.HTML("<div class='settings-section-title'>Background Results</div>")
                scheduled_results_md = gr.Markdown(
                    _scheduled_results_summary_md(),
                    elem_id="scheduled-results-summary",
                )
                scheduled_result_picker = gr.Dropdown(
                    label="Recent Background Jobs",
                    choices=_scheduled_result_choices(),
                    value=None,
                    allow_custom_value=False,
                )
                with gr.Row(elem_id="scheduled-result-actions"):
                    load_scheduled_result_btn = gr.Button("Load Result", elem_classes=["secondary-btn"])
                    load_scheduled_log_btn = gr.Button("Load Log", elem_classes=["secondary-btn"])
                rerun_scheduled_btn = gr.Button("Re-run Selected", elem_classes=["secondary-btn"])
                scheduled_result_viewer = gr.Markdown(
                    "No scheduled result selected.",
                    elem_id="scheduled-result-viewer",
                )
                refresh_timer = gr.Timer(5)

        settings_inputs = [base_url_in, model_in, max_steps_in]
        history_outputs = [history_md] + history_buttons + history_task_states

        def _full_history_outputs() -> list[Any]:
            return _history_control_updates()

        for chip_button, (_, task_text) in zip(chip_buttons, EXAMPLE_TASKS):
            chip_button.click(lambda text=task_text: text, outputs=[msg])

        for history_button, history_state in zip(history_buttons, history_task_states):
            history_button.click(lambda text: text, inputs=[history_state], outputs=[msg])

        async def queue_task_action(message, chat_history, base_url, model, max_steps, *slot_task_ids_values):
            """Add a task to the background queue. Result will flow back into chat via the refresh timer."""
            chat_history = chat_history or []
            task_text = message.strip()
            if not task_text:
                return gr.update(), chat_history, _queue_summary_md(), _queued_task_dropdown_update()

            await _ensure_scheduler_started()
            item = QueuedTask(
                id=uuid4().hex,
                task=task_text,
                base_url=base_url,
                model=model,
                max_steps=_normalize_max_steps(max_steps),
                created_at=datetime.now(),
                scheduled_for=None,
                source="queued",
            )
            await _enqueue_task(item)

            position = len(task_queue)
            chat_history.append({"role": "user", "content": task_text})
            chat_history.append({
                "role": "assistant",
                "content": (
                    f"Browser task **{_short_task_text(task_text, 120)}** added to queue "
                    f"(position {position}). Result will appear here when complete."
                ),
            })
            return "", chat_history, _queue_summary_md(), _queued_task_dropdown_update()

        async def respond(message, chat_history, base_url, model, max_steps):
            history_bundle = _full_history_outputs()
            scheduled_summary, scheduled_picker_update = _scheduled_panel_updates()
            if not message.strip():
                yield (
                    gr.update(),
                    _drain_pending_chat_messages(chat_history),
                    gr.update(),
                    gr.update(),
                    _steps_md([]),
                    "<div id='step-counter'>0</div>",
                    _tabs_md([]),
                    history_bundle[0],
                    _queue_summary_md(),
                    _queued_task_dropdown_update(),
                    gr.update(),
                    scheduled_summary,
                    scheduled_picker_update,
                    gr.update(),
                    gr.update(),
                    "",
                    "",
                    [],
                    "",
                    *history_bundle[1:],
                )
                return

            await _ensure_scheduler_started()
            chat_history = _drain_pending_chat_messages(chat_history)

            if runner_lock.locked():
                item = QueuedTask(
                    id=uuid4().hex,
                    task=message,
                    base_url=base_url,
                    model=model,
                    max_steps=_normalize_max_steps(max_steps),
                    created_at=datetime.now(),
                    scheduled_for=None,
                    source="queued",
                )
                await _enqueue_task(item)
                position = len(task_queue)
                chat_history.append({"role": "user", "content": message})
                chat_history.append({
                    "role": "assistant",
                    "content": (
                        f"Browser task **{_short_task_text(message, 120)}** added to queue "
                        f"(position {position}). Result will appear here when complete."
                    ),
                })
                history_bundle = _full_history_outputs()
                scheduled_summary, scheduled_picker_update = _scheduled_panel_updates()
                yield (
                    "",
                    chat_history,
                    gr.update(), gr.update(),
                    gr.update(), gr.update(),
                    gr.update(),
                    history_bundle[0],
                    _queue_summary_md(),
                    _queued_task_dropdown_update(),
                    gr.update(), scheduled_summary, scheduled_picker_update,
                    gr.update(), gr.update(),
                    message, "", [], "",
                    *history_bundle[1:],
                )
                return

            chat_history.append({"role": "user", "content": message})
            chat_history.append({
                "role": "assistant",
                "content": "Queued. Waiting for the browser worker to pick this task up.",
            })

            item = QueuedTask(
                id=uuid4().hex,
                task=message,
                base_url=base_url,
                model=model,
                max_steps=_normalize_max_steps(max_steps),
                created_at=datetime.now(),
                update_queue=asyncio.Queue(),
            )
            await _enqueue_task(item)

            yield (
                "",
                chat_history,
                gr.update(),
                gr.update(),
                _steps_md([]),
                "<div id='step-counter'>0</div>",
                _tabs_md([]),
                history_bundle[0],
                _queue_summary_md(),
                _queued_task_dropdown_update(),
                gr.update(),
                scheduled_summary,
                scheduled_picker_update,
                gr.update(),
                gr.update(value=None, visible=False),
                message,
                "",
                [],
                "",
                *history_bundle[1:],
            )

            step_messages: list[str] = []
            latest_tabs: list[dict] = []
            payload: dict[str, Any] | None = None
            _last_status_text: str = ""  # track last timeout-yielded status to skip duplicate re-renders

            while payload is None:
                try:
                    event = await asyncio.wait_for(item.update_queue.get(), timeout=QUEUE_POLL_SECONDS)
                except asyncio.TimeoutError:
                    position = await _queue_position(item.id)
                    if position is not None:
                        new_status = f"Queued. Position in line: {position}."
                    else:
                        new_status = "Working on the request..."

                    has_pending = bool(_pending_chat_messages)
                    if new_status == _last_status_text and not has_pending:
                        # Nothing changed — skip the yield entirely to avoid chatbot blink.
                        continue
                    _last_status_text = new_status
                    chat_history[-1] = {"role": "assistant", "content": new_status}
                    chat_history = _drain_pending_chat_messages(chat_history)
                    history_bundle = _full_history_outputs()
                    scheduled_summary, scheduled_picker_update = _scheduled_panel_updates()
                    yield (
                        gr.update(),
                        chat_history,
                        gr.update(),
                        gr.update(),
                        _steps_md(step_messages),
                        f"<div id='step-counter'>{len(step_messages)}</div>",
                        _tabs_md(latest_tabs),
                        history_bundle[0],
                        _queue_summary_md(),
                        _queued_task_dropdown_update(),
                        gr.update(),
                        scheduled_summary,
                        scheduled_picker_update,
                        gr.update(),
                        gr.update(value=None, visible=False),
                        message,
                        "",
                        latest_tabs,
                        "",
                        *history_bundle[1:],
                    )
                    continue

                if event["type"] == "started":
                    chat_history[-1] = {"role": "assistant", "content": "Working..."}
                elif event["type"] == "step":
                    step_messages.append(event["message"])
                    latest_tabs = event.get("tabs", [])
                    chat_history[-1] = {
                        "role": "assistant",
                        "content": f"Working...\n\nLatest step: {event['message']}",
                    }
                elif event["type"] == "completed":
                    payload = event["payload"]
                    break

                chat_history = _drain_pending_chat_messages(chat_history)
                history_bundle = _full_history_outputs()
                scheduled_summary, scheduled_picker_update = _scheduled_panel_updates()
                yield (
                    gr.update(),
                    chat_history,
                    gr.update(),
                    gr.update(),
                    _steps_md(step_messages),
                    f"<div id='step-counter'>{len(step_messages)}</div>",
                    _tabs_md(latest_tabs),
                    history_bundle[0],
                    _queue_summary_md(),
                    _queued_task_dropdown_update(),
                    gr.update(),
                    scheduled_summary,
                    scheduled_picker_update,
                    gr.update(),
                    gr.update(value=None, visible=False),
                    message,
                    "",
                    latest_tabs,
                    "",
                    *history_bundle[1:],
                )

            chat_history = _drain_pending_chat_messages(chat_history)
            history_bundle = payload["history_controls"]
            scheduled_summary, scheduled_picker_update = payload["scheduled_panel"]
            if payload["status"] == "completed":
                response_text = (
                    f"**Result:**\n\n{payload['result_text']}\n\n---\n"
                    "Browser remains open for verification."
                    f"\n\nTroubleshooting log saved to `{Path(payload['log_path']).name}`."
                )
                if payload.get("screenshot_path"):
                    response_text += f"\nScreenshot saved to `{payload['screenshot_path']}`."
                if payload.get("screenshot"):
                    response_text += "\n\nScreenshot captured in the right panel."
                    pil_image = PILImage.open(io.BytesIO(payload["screenshot"]))
                    screenshot_update = gr.update(value=pil_image, visible=True)
                    screenshot_hint_update = gr.update(visible=False)
                else:
                    screenshot_update = gr.update(value=None, visible=False)
                    screenshot_hint_update = gr.update(visible=True)
                chat_history[-1] = {"role": "assistant", "content": response_text}
                last_result_value = payload["result_text"]
                last_tabs_value = payload["tabs"]
            else:
                error_text = payload["error_text"]
                if "refused" in error_text.lower() or "connection" in error_text.lower():
                    response_text = "Cannot reach LM Studio. Make sure the server is running on the configured URL."
                else:
                    response_text = f"Error: {error_text}"
                response_text += f"\n\nTroubleshooting log saved to `{Path(payload['log_path']).name}`."
                chat_history[-1] = {"role": "assistant", "content": response_text}
                screenshot_update = gr.update(value=None, visible=False)
                screenshot_hint_update = gr.update(visible=True)
                last_result_value = ""
                last_tabs_value = payload.get("tabs", latest_tabs)

            yield (
                gr.update(),
                chat_history,
                screenshot_update,
                screenshot_hint_update,
                payload["steps_md"],
                f"<div id='step-counter'>{payload['step_count']}</div>",
                _tabs_md(last_tabs_value),
                history_bundle[0],
                _queue_summary_md(),
                _queued_task_dropdown_update(),
                load_latest_log(),
                scheduled_summary,
                scheduled_picker_update,
                _scheduled_result_view(payload["task_id"]) if payload.get("task_id") else gr.update(),
                gr.update(value=None, visible=False),
                message,
                last_result_value,
                last_tabs_value,
                payload["log_path"],
                *history_bundle[1:],
            )

        async def schedule_task(message, schedule_date, schedule_hr, schedule_min, schedule_ampm, chat_history, base_url, model, max_steps):
            chat_history = chat_history or []
            task_text = message.strip()
            if not task_text:
                chat_history.append({"role": "assistant", "content": "Enter a task before scheduling it."})
                return message, chat_history, _queue_summary_md(), _queued_task_dropdown_update(), *_scheduled_panel_updates()

            # Build scheduled_for from calendar date + three time dropdowns
            scheduled_for: datetime | None = None
            if schedule_date:
                try:
                    date_str = str(schedule_date).strip()[:10]  # "YYYY-MM-DD"
                    hr = int(schedule_hr or 12)
                    mn = int(schedule_min or 0)
                    if (schedule_ampm or "AM").upper() == "PM" and hr != 12:
                        hr += 12
                    elif (schedule_ampm or "AM").upper() == "AM" and hr == 12:
                        hr = 0
                    scheduled_for = datetime.strptime(f"{date_str} {hr:02d}:{mn:02d}", "%Y-%m-%d %H:%M")
                except (ValueError, TypeError) as exc:
                    chat_history.append({"role": "assistant", "content": f"Could not parse date/time: {exc}"})
                    return message, chat_history, _queue_summary_md(), _queued_task_dropdown_update(), *_scheduled_panel_updates()

            if scheduled_for and scheduled_for < datetime.now():
                chat_history.append({"role": "assistant", "content": "Scheduled time must be in the future."})
                return message, chat_history, _queue_summary_md(), _queued_task_dropdown_update(), *_scheduled_panel_updates()

            await _ensure_scheduler_started()
            await _enqueue_task(
                QueuedTask(
                    id=uuid4().hex,
                    task=task_text,
                    base_url=base_url,
                    model=model,
                    max_steps=_normalize_max_steps(max_steps),
                    created_at=datetime.now(),
                    scheduled_for=scheduled_for,
                    source="scheduled",
                )
            )

            if scheduled_for is None:
                when_text = "next available slot"
            else:
                when_text = scheduled_for.strftime("%Y-%m-%d %H:%M")

            chat_history.append(
                {
                    "role": "assistant",
                    "content": f"Scheduled task for {when_text}: {_short_task_text(task_text, 120)}",
                }
            )
            scheduled_summary, scheduled_picker_update = _scheduled_panel_updates()
            return "", chat_history, _queue_summary_md(), _queued_task_dropdown_update(), scheduled_summary, scheduled_picker_update

        async def cancel_selected_queue_item(selected_task_id):
            if not selected_task_id:
                scheduled_summary, scheduled_picker_update = _scheduled_panel_updates()
                return _queue_summary_md(), _queued_task_dropdown_update(), scheduled_summary, scheduled_picker_update

            await _remove_queued_task(selected_task_id)
            scheduled_summary, scheduled_picker_update = _scheduled_panel_updates()
            return _queue_summary_md(), _queued_task_dropdown_update(), scheduled_summary, scheduled_picker_update

        def load_scheduled_result(task_id):
            return _scheduled_result_view(task_id)

        def load_scheduled_log(task_id):
            record = _get_scheduled_record(task_id)
            if not record or not record.get("log_path"):
                return "No log is available for the selected scheduled job."
            log_path = Path(record["log_path"])
            if not log_path.exists():
                return f"Saved log file not found: `{log_path}`"
            try:
                data = json.loads(log_path.read_text(encoding="utf-8"))
            except Exception as exc:
                return f"Could not read scheduled log: {exc}"

            lines = [
                f"**File:** `{log_path.name}`",
                f"**Task:** {data.get('task', '').strip()}",
                f"**Status:** {data.get('status', '').upper()}",
                f"**Started:** {data.get('started_at', '')}",
                f"**Duration:** {data.get('duration_seconds', '')}s",
            ]
            if data.get("result"):
                lines += ["", "**Result:**", data["result"]]
            if data.get("error"):
                lines += ["", "**Error:**", data["error"]]
            return "\n".join(lines)

        def rerun_selected_scheduled(task_id):
            record = _get_scheduled_record(task_id)
            if not record:
                return gr.update(), None
            return record.get("task", ""), None

        async def close_browser(chat_history, base_url, model, max_steps):
            session = get_session(base_url, model, max_steps)
            chat_history = chat_history or []
            if session.is_browser_open:
                await session.close_browser()
                chat_history.append({"role": "assistant", "content": "Browser closed."})
            else:
                chat_history.append({"role": "assistant", "content": "No browser is currently open."})
            return chat_history

        def clear_chat():
            _pending_chat_messages.clear()
            return (
                [],
                gr.update(value=None, visible=False),
                gr.update(visible=True),
                _steps_md([]),
                "<div id='step-counter'>0</div>",
                _tabs_md([]),
                gr.update(value=None, visible=False),
                "",
                "",
                [],
                "",
            )

        async def apply_settings(base_url, model, max_steps):
            global agent_session, agent_session_config
            normalized_max_steps = _normalize_max_steps(max_steps)
            _retarget_pending_tasks(base_url, model, normalized_max_steps)
            if agent_session and agent_session.is_browser_open:
                await agent_session.close_browser()
            agent_session = build_session(base_url, model, normalized_max_steps)
            agent_session_config = (base_url, model, normalized_max_steps)
            return _status_md(base_url, model, normalized_max_steps)

        def refresh_status(base_url, model, max_steps):
            return _status_md(base_url, model, max_steps)

        def refresh_side_panels(chat_history):
            # Only update chatbot when there are actually pending background messages.
            # Returning gr.update() is a no-op that prevents Gradio from re-rendering
            # the chatbot on every tick, which caused the visible 5-second blink.
            if _pending_chat_messages:
                chatbot_update = _drain_pending_chat_messages(chat_history)
            else:
                chatbot_update = gr.update()
            history_bundle = _full_history_outputs()
            scheduled_summary, scheduled_picker_update = _scheduled_panel_updates()
            return (
                chatbot_update,
                history_bundle[0],
                _queue_summary_md(),
                _queued_task_dropdown_update(),
                scheduled_summary,
                scheduled_picker_update,
                *history_bundle[1:],
            )

        respond_outputs = [
            msg,
            chatbot,
            screenshot_img,
            screenshot_hint,
            live_steps_md,
            step_counter,
            tabs_md,
            history_md,
            queue_md,
            queue_task_picker,
            log_viewer,
            scheduled_results_md,
            scheduled_result_picker,
            scheduled_result_viewer,
            export_file,
            last_task_state,
            last_result_state,
            last_tabs_state,
            last_log_path_state,
            *history_buttons,
            *history_task_states,
        ]

        msg.submit(respond, [msg, chatbot] + settings_inputs, respond_outputs)
        send_btn.click(respond, [msg, chatbot] + settings_inputs, respond_outputs)
        queue_btn.click(
            queue_task_action,
            inputs=[msg, chatbot] + settings_inputs,
            outputs=[msg, chatbot, queue_md, queue_task_picker],
        )
        schedule_btn.click(
            schedule_task,
            inputs=[msg, schedule_date_in, schedule_hr_in, schedule_min_in, schedule_ampm_in, chatbot] + settings_inputs,
            outputs=[msg, chatbot, queue_md, queue_task_picker, scheduled_results_md, scheduled_result_picker],
        )
        cancel_queue_btn.click(
            cancel_selected_queue_item,
            inputs=[queue_task_picker],
            outputs=[queue_md, queue_task_picker, scheduled_results_md, scheduled_result_picker],
        )
        close_btn.click(close_browser, [chatbot] + settings_inputs, [chatbot])
        clear_btn.click(
            clear_chat,
            outputs=[
                chatbot,
                screenshot_img,
                screenshot_hint,
                live_steps_md,
                step_counter,
                tabs_md,
                export_file,
                last_task_state,
                last_result_state,
                last_tabs_state,
                last_log_path_state,
            ],
        )
        apply_btn.click(apply_settings, settings_inputs, [status_md])
        refresh_status_btn.click(refresh_status, settings_inputs, [status_md])
        export_btn.click(
            export_last_result,
            inputs=[last_task_state, last_result_state, last_tabs_state],
            outputs=[export_file],
        )
        open_log_btn.click(lambda: load_latest_log(), outputs=[log_viewer])
        load_scheduled_result_btn.click(
            load_scheduled_result,
            inputs=[scheduled_result_picker],
            outputs=[scheduled_result_viewer],
        )
        load_scheduled_log_btn.click(
            load_scheduled_log,
            inputs=[scheduled_result_picker],
            outputs=[log_viewer],
        )
        rerun_scheduled_btn.click(
            rerun_selected_scheduled,
            inputs=[scheduled_result_picker],
            outputs=[msg, schedule_date_in],
        )
        refresh_timer.tick(
            refresh_side_panels,
            inputs=[chatbot],
            outputs=[chatbot, history_md, queue_md, queue_task_picker, scheduled_results_md, scheduled_result_picker,
                     *history_buttons, *history_task_states],
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    server_port = _find_available_port()
    print(f"Starting Browser Pilot on http://127.0.0.1:{server_port}")
    app.launch(
        server_name="127.0.0.1",
        server_port=server_port,
        share=False,
        css=CSS,
        js=JS,
    )
