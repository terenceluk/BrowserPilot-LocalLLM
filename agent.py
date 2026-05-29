"""
Browser agent powered by browser-use + LM Studio.
"""

import asyncio
import inspect
import json
import os
import sys
from collections.abc import Iterable

from openai import APIConnectionError, APIStatusError, RateLimitError
from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.shared_params.response_format_json_schema import JSONSchema, ResponseFormatJSONSchema

import config as cfg
from browser_use.llm.models import ChatOpenAI
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.openai.serializer import OpenAIMessageSerializer
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion
from browser_use.agent.service import Agent
from browser_use.agent.views import ActionResult
from browser_use.browser import BrowserProfile, BrowserSession
from browser_use.tools.service import Tools
from config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
    MODEL_NAME,
    BROWSER_HEADLESS,
    MAX_STEPS,
    USE_VISION,
    BROWSER_WIDTH,
    BROWSER_HEIGHT,
)
from pydantic import BaseModel, Field


class CheckboxByLabelAction(BaseModel):
        label_text: str = Field(description="Visible text next to or above the checkbox to toggle")
        checked: bool = Field(default=True, description="Desired checkbox state")
        exact_match: bool = Field(default=False, description="Require exact text match instead of contains match")


class InputByLabelAction(BaseModel):
    label_text: str = Field(description="Visible field label, placeholder, aria-label, or name")
    value: str = Field(description="Text value to place into the field")
    exact_match: bool = Field(default=False, description="Require exact text match instead of contains match")


def build_agent_tools() -> Tools:
        tools = Tools()

        class ClickElementByTextAction(BaseModel):
            text: str = Field(description="Visible text or title of the button/link to click")
            exact_match: bool = Field(default=False, description="Require exact text match instead of contains match")
            tag: str = Field(default="any", description="Restrict to tag: 'button', 'a', 'div', 'span', or 'any'")

        @tools.action(
            "Click a button, link, or clickable element by visible text or title. Use this when index-based click fails or for elements inside shadow DOM.",
            param_model=ClickElementByTextAction,
        )
        async def click_element_by_text(params: ClickElementByTextAction, browser_session: BrowserSession):
            cdp_session = await browser_session.get_or_create_cdp_session()
            target_text = params.text.strip()
            exact_match = params.exact_match
            tag = params.tag.lower()

            js_code = r"""(function(targetText, exactMatch, tag) {
    const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim().toLowerCase();
    const target = normalize(targetText);
    if (!target) return { ok: false, error: 'Empty target text.' };

    const tags = tag === 'any' ? ['button', 'a', 'div', 'span'] : [tag];
    const isVisible = (el) => {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
    };

    const collectRoots = () => {
        const roots = [document];
        const seenNodes = new Set();
        const queue = [document.documentElement];
        while (queue.length > 0) {
            const node = queue.shift();
            if (!node || seenNodes.has(node)) continue;
            seenNodes.add(node);
            if (node.shadowRoot && !roots.includes(node.shadowRoot)) {
                roots.push(node.shadowRoot);
                for (const child of Array.from(node.shadowRoot.children || [])) queue.push(child);
            }
            for (const child of Array.from(node.children || [])) queue.push(child);
        }
        return roots;
    };

    const queryAllDeep = (selector) => {
        const results = [];
        const seen = new Set();
        for (const root of collectRoots()) {
            for (const el of Array.from(root.querySelectorAll(selector))) {
                if (seen.has(el)) continue;
                seen.add(el);
                results.push(el);
            }
        }
        return results;
    };

    let candidates = [];
    for (const tagName of tags) {
        const selector = tagName;
        for (const el of queryAllDeep(selector)) {
            if (!isVisible(el)) continue;
            const texts = [el.textContent, el.getAttribute('title'), el.getAttribute('aria-label')].filter(Boolean);
            for (const text of texts) {
                const norm = normalize(text);
                if ((exactMatch && norm === target) || (!exactMatch && norm.includes(target))) {
                    candidates.push({ el, text });
                    break;
                }
            }
        }
    }
    if (!candidates.length) {
        return { ok: false, error: `No element found with text/title: ${targetText}` };
    }
    // Prefer exact match, then first candidate
    let match = candidates.find(c => normalize(c.text) === target) || candidates[0];
    match.el.scrollIntoView({ block: 'center', inline: 'center' });
    match.el.click();
    return { ok: true, clicked_text: match.text };
})(targetText, exactMatch, tag);
"""

            result = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={
                    "expression": f"{js_code}(" +
                    f"{json.dumps(target_text)}, {json.dumps(exact_match)}, {json.dumps(tag)})",
                    "returnByValue": True,
                    "awaitPromise": True,
                },
                session_id=cdp_session.session_id,
            )

            if result.get("exceptionDetails"):
                error = result["exceptionDetails"].get("text", "JavaScript execution failed")
                return ActionResult(error=f"click_element_by_text failed: {error}")

            payload = result.get("result", {}).get("value") or {}
            if not payload.get("ok"):
                return ActionResult(error=payload.get("error", "Element was not clicked"))

            return ActionResult(
                extracted_content=f"Clicked element with text/title '{payload.get('clicked_text', target_text)}'",
                long_term_memory=f"Clicked element near '{target_text}'",
            )

        @tools.action(
                "Set a checkbox by matching nearby visible label text. Use this when a checkbox is custom-styled or repeated click(index) fails. After calling it, verify the checkbox state before submitting a form.",
                param_model=CheckboxByLabelAction,
        )
        async def set_checkbox_by_label(params: CheckboxByLabelAction, browser_session: BrowserSession):
                cdp_session = await browser_session.get_or_create_cdp_session()
                target_label = params.label_text.strip()
                desired_state = params.checked
                exact_match = params.exact_match

                js_code = r"""(function(labelText, desiredState, exactMatch) {
    const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim().toLowerCase();
    const target = normalize(labelText);
    if (!target) {
        return { ok: false, error: 'Empty label text.' };
    }

    const checkboxSelector = 'input[type="checkbox"]';
    const textContainerSelector = 'label, span, div, p, strong, b, td, li';

    const isVisible = (element) => {
        if (!element) return false;
        const style = window.getComputedStyle(element);
        if (style.display === 'none' || style.visibility === 'hidden') return false;
        const rect = element.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
    };

    const textMatches = (text) => {
        const normalized = normalize(text);
        return exactMatch ? normalized === target : normalized.includes(target);
    };

    const getCheckboxText = (checkbox) => {
        const checks = [
            checkbox.getAttribute('placeholder'),
            checkbox.getAttribute('aria-label'),
            checkbox.name,
            checkbox.id,
        ].filter(Boolean);

        const owningLabel = checkbox.closest('label');
        if (owningLabel) {
            checks.push(owningLabel.textContent || '');
        }

        if (checkbox.id) {
            const explicitLabel = document.querySelector(`label[for="${CSS.escape(checkbox.id)}"]`);
            if (explicitLabel) {
                checks.push(explicitLabel.textContent || '');
            }
        }

        return checks;
    };

    const addCandidate = (candidates, seen, checkbox, matchedText, score) => {
        if (!checkbox || seen.has(checkbox) || !isVisible(checkbox)) return;
        seen.add(checkbox);
        candidates.push({ checkbox, text: (matchedText || '').trim(), score });
    };

    const findCheckboxNear = (element) => {
        if (!element) return null;
        if (element.matches?.(checkboxSelector)) return element;

        if (element.matches?.('label')) {
            const forId = element.getAttribute('for');
            if (forId) {
                const linked = document.getElementById(forId);
                if (linked?.matches?.(checkboxSelector)) return linked;
            }

            const nested = element.querySelector(checkboxSelector);
            if (nested) return nested;
        }

        const closestLabel = element.closest?.('label');
        if (closestLabel) {
            const forId = closestLabel.getAttribute('for');
            if (forId) {
                const linked = document.getElementById(forId);
                if (linked?.matches?.(checkboxSelector)) return linked;
            }

            const nested = closestLabel.querySelector(checkboxSelector);
            if (nested) return nested;
        }

        const searchRoots = [
            element.parentElement,
            element.previousElementSibling,
            element.nextElementSibling,
            element.closest?.('div, li, td, p, section, article, form, fieldset'),
        ].filter(Boolean);

        for (const root of searchRoots) {
            if (root?.matches?.(checkboxSelector)) return root;
            const nested = root?.querySelector?.(checkboxSelector);
            if (nested) return nested;
        }

        return null;
    };

    const candidates = [];
    const seen = new Set();

    for (const checkbox of Array.from(document.querySelectorAll(checkboxSelector))) {
        const checks = getCheckboxText(checkbox);
        const matched = checks.find(textMatches);
        if (!matched) continue;

        let score = 10;
        const explicitLabel = checkbox.id
            ? document.querySelector(`label[for="${CSS.escape(checkbox.id)}"]`)
            : null;
        if (explicitLabel && textMatches(explicitLabel.textContent || '')) {
            score = 40;
        } else if ((checkbox.getAttribute('placeholder') || checkbox.getAttribute('aria-label')) && matched === (checkbox.getAttribute('placeholder') || checkbox.getAttribute('aria-label'))) {
            score = 35;
        } else if (checkbox.closest('label') && textMatches(checkbox.closest('label').textContent || '')) {
            score = 30;
        }

        if (normalize(matched) === target) {
            score += 10;
        }

        addCandidate(candidates, seen, checkbox, matched, score);
    }

    for (const element of Array.from(document.querySelectorAll(textContainerSelector))) {
        if (!isVisible(element)) continue;
        const text = element.textContent || '';
        if (!textMatches(text)) continue;

        const checkbox = findCheckboxNear(element);
        if (!checkbox) continue;

        const exact = normalize(text) === target;
        const score = element.matches?.('label[for]') ? (exact ? 40 : 36) : (exact ? 24 : 18);
        addCandidate(candidates, seen, checkbox, text, score);
    }

    candidates.sort((left, right) => right.score - left.score);
    const match = candidates[0];
    if (!match) {
        return { ok: false, error: `No checkbox found near label text: ${labelText}` };
    }

    const checkbox = match.checkbox;
    checkbox.scrollIntoView({ block: 'center', inline: 'center' });

    const dispatchStateEvents = () => {
        checkbox.dispatchEvent(new Event('input', { bubbles: true }));
        checkbox.dispatchEvent(new Event('change', { bubbles: true }));
        checkbox.dispatchEvent(new Event('blur', { bubbles: true }));
    };

    const clickCustomControl = () => {
        const customLabel = checkbox.parentElement?.querySelector('.custom-label');
        if (customLabel && isVisible(customLabel)) {
            customLabel.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window }));
            return;
        }

        const explicitLabel = checkbox.id
            ? document.querySelector(`label[for="${CSS.escape(checkbox.id)}"]`)
            : null;
        if (explicitLabel && isVisible(explicitLabel)) {
            explicitLabel.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window }));
        }
    };

    const applyState = (value) => {
        if (checkbox.checked !== value) {
            checkbox.click();
        }

        if (checkbox.checked !== value) {
            checkbox.checked = value;
            dispatchStateEvents();
        }

        if (checkbox.checked !== value) {
            clickCustomControl();
        }

        if (checkbox.checked !== value) {
            checkbox.checked = value;
            dispatchStateEvents();
        }
    };

    applyState(desiredState);

    return {
        ok: checkbox.checked === desiredState,
        error: checkbox.checked === desiredState ? null : `Checkbox state verification failed for label: ${labelText}`,
        checked: checkbox.checked,
        desired: desiredState,
        matched_text: match.text,
    };
})"""

                result = await cdp_session.cdp_client.send.Runtime.evaluate(
                        params={
                                "expression": f"{js_code}({json.dumps(target_label)}, {json.dumps(desired_state)}, {json.dumps(exact_match)})",
                                "returnByValue": True,
                                "awaitPromise": True,
                        },
                        session_id=cdp_session.session_id,
                )

                if result.get("exceptionDetails"):
                        error = result["exceptionDetails"].get("text", "JavaScript execution failed")
                        return ActionResult(error=f"set_checkbox_by_label failed: {error}")

                payload = result.get("result", {}).get("value") or {}
                if not payload.get("ok"):
                        return ActionResult(error=payload.get("error", "Checkbox state was not updated"))

                return ActionResult(
                        extracted_content=(
                                f"Checkbox '{payload.get('matched_text', target_label)}' set to {payload.get('checked')}"
                        ),
                        long_term_memory=(
                                f"Checkbox near '{target_label}' now {payload.get('checked')}"
                        ),
                )

        @tools.action(
                "Set an input, email, password, or textarea value by matching a nearby label, placeholder, aria-label, or field name. Use this when typing retries append duplicate text or a login field must be filled reliably. Verify the final field value before submitting.",
                param_model=InputByLabelAction,
        )
        async def set_input_value_by_label(params: InputByLabelAction, browser_session: BrowserSession):
                cdp_session = await browser_session.get_or_create_cdp_session()
                target_label = params.label_text.strip()
                target_value = params.value
                exact_match = params.exact_match

                js_code = r"""(function(labelText, desiredValue, exactMatch) {
    const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim().toLowerCase();
    const target = normalize(labelText);
    if (!target) {
        return { ok: false, error: 'Empty field label.' };
    }

    const fieldSelector = 'input:not([type="hidden"]):not([type="checkbox"]):not([type="radio"]):not([type="submit"]):not([type="button"]), textarea';
    const textContainerSelector = 'label, span, div, p, strong, b, td, li';

    const textMatches = (text) => {
        const normalized = normalize(text);
        return exactMatch ? normalized === target : normalized.includes(target);
    };

    const getExpectedTypes = () => {
        if (target.includes('password')) return new Set(['password']);
        if (target.includes('email') || target.includes('e-mail')) return new Set(['email']);
        return null;
    };

    const expectedTypes = getExpectedTypes();

    const collectRoots = () => {
        const roots = [document];
        const seenNodes = new Set();
        const queue = [document.documentElement];

        while (queue.length > 0) {
            const node = queue.shift();
            if (!node || seenNodes.has(node)) continue;
            seenNodes.add(node);

            if (node.shadowRoot && !roots.includes(node.shadowRoot)) {
                roots.push(node.shadowRoot);
                for (const child of Array.from(node.shadowRoot.children || [])) {
                    queue.push(child);
                }
            }

            for (const child of Array.from(node.children || [])) {
                queue.push(child);
            }
        }

        return roots;
    };

    const queryAllDeep = (selector) => {
        const results = [];
        const seen = new Set();

        for (const root of collectRoots()) {
            for (const element of Array.from(root.querySelectorAll(selector))) {
                if (seen.has(element)) continue;
                seen.add(element);
                results.push(element);
            }
        }

        return results;
    };

    const isVisible = (element) => {
        if (!element) return false;
        const style = window.getComputedStyle(element);
        if (style.display === 'none' || style.visibility === 'hidden') return false;
        const rect = element.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
    };

    const getFieldType = (field) => {
        if (!field) return '';
        if (field instanceof HTMLTextAreaElement) return 'textarea';
        return normalize(field.type || field.tagName);
    };

    const fieldTypeMatches = (field) => {
        if (!expectedTypes) return true;
        return expectedTypes.has(getFieldType(field));
    };

    const scoreField = (field) => {
        const checks = [
            field.placeholder,
            field.getAttribute('aria-label'),
            field.name,
            field.id,
            field.type,
        ].filter(Boolean);
        let score = checks.some(textMatches) ? 2 : 0;
        if (expectedTypes) {
            score += fieldTypeMatches(field) ? 6 : -8;
        }
        return score;
    };

    const findFieldNear = (element) => {
        if (!element) return null;
        if (element.matches?.(fieldSelector)) return element;

        if (element.matches?.('label')) {
            const nested = element.querySelector(fieldSelector);
            if (nested) return nested;
            const forId = element.getAttribute('for');
            if (forId) {
                const linked = document.getElementById(forId);
                if (linked?.matches?.(fieldSelector)) return linked;
            }
        }

        const closestLabel = element.closest?.('label');
        if (closestLabel) {
            const nested = closestLabel.querySelector(fieldSelector);
            if (nested) return nested;
            const forId = closestLabel.getAttribute('for');
            if (forId) {
                const linked = document.getElementById(forId);
                if (linked?.matches?.(fieldSelector)) return linked;
            }
        }

        const roots = [
            element.parentElement,
            element.previousElementSibling,
            element.nextElementSibling,
            element.closest?.('div, li, td, p, section, article, form, fieldset'),
        ].filter(Boolean);

        for (const root of roots) {
            if (root?.matches?.(fieldSelector)) return root;
            const nested = root?.querySelector?.(fieldSelector);
            if (nested) return nested;

            const nestedDeep = queryAllDeep(fieldSelector).find((field) => root?.contains?.(field));
            if (nestedDeep) return nestedDeep;
        }

        return null;
    };

    const candidates = [];
    const seen = new Set();

    for (const field of queryAllDeep(fieldSelector)) {
        if (!isVisible(field)) continue;
        const checks = [field.placeholder, field.getAttribute('aria-label'), field.name, field.id].filter(Boolean);
        if (!checks.some(textMatches)) continue;
        if (seen.has(field)) continue;
        seen.add(field);
        candidates.push({
            field,
            score: 3 + scoreField(field),
            matched_text: checks.find(textMatches) || field.name || field.id || '',
        });
    }

    for (const element of queryAllDeep(textContainerSelector)) {
        if (!isVisible(element)) continue;
        const text = element.textContent || '';
        if (!textMatches(text)) continue;
        const field = findFieldNear(element);
        if (!field || seen.has(field)) continue;
        seen.add(field);
        const exact = normalize(text) === target;
        candidates.push({ field, score: exact ? 5 : 4 + scoreField(field), matched_text: text.trim() });
    }

    candidates.sort((left, right) => right.score - left.score);
    const match = candidates[0];
    if (!match) {
        return { ok: false, error: `No input found near label text: ${labelText}` };
    }

    const field = match.field;
    if (!fieldTypeMatches(field)) {
        return {
            ok: false,
            error: `Matched an incompatible field type '${getFieldType(field)}' for label: ${labelText}`,
            matched_text: match.matched_text,
            input_type: getFieldType(field),
        };
    }

    field.scrollIntoView({ block: 'center', inline: 'center' });
    field.focus();

    const prototype = field instanceof HTMLTextAreaElement
        ? HTMLTextAreaElement.prototype
        : HTMLInputElement.prototype;
    const descriptor = Object.getOwnPropertyDescriptor(prototype, 'value');
    if (descriptor?.set) {
        descriptor.set.call(field, desiredValue);
    } else {
        field.value = desiredValue;
    }

    field.dispatchEvent(new Event('input', { bubbles: true }));
    field.dispatchEvent(new Event('change', { bubbles: true }));
    field.dispatchEvent(new Event('blur', { bubbles: true }));

    return {
        ok: field.value === desiredValue,
        error: field.value === desiredValue ? null : `Field value verification failed for label: ${labelText}`,
        matched_text: match.matched_text,
        actual_value: field.value,
        input_type: getFieldType(field),
    };
})"""

                result = await cdp_session.cdp_client.send.Runtime.evaluate(
                        params={
                                "expression": f"{js_code}({json.dumps(target_label)}, {json.dumps(target_value)}, {json.dumps(exact_match)})",
                                "returnByValue": True,
                                "awaitPromise": True,
                        },
                        session_id=cdp_session.session_id,
                )

                if result.get("exceptionDetails"):
                        error = result["exceptionDetails"].get("text", "JavaScript execution failed")
                        return ActionResult(error=f"set_input_value_by_label failed: {error}")

                payload = result.get("result", {}).get("value") or {}
                if not payload.get("ok"):
                        return ActionResult(error=payload.get("error", "Input value was not updated"))

                input_type = str(payload.get("input_type", "")).lower()
                matched_text = payload.get("matched_text", target_label)
                if input_type == "password" or "password" in matched_text.lower():
                        summary = f"Input '{matched_text}' updated"
                else:
                        summary = f"Input '{matched_text}' set to '{payload.get('actual_value', '')}'"

                return ActionResult(
                        extracted_content=summary,
                        long_term_memory=f"Field near '{target_label}' updated",
                )

        return tools


class ConsoleTracingChatOpenAI(ChatOpenAI):
    _ANSI_RESET = "\033[0m"
    _ANSI_BOLD = "\033[1m"
    _SECTION_COLORS = {
        "meta": "\033[38;5;81m",
        "content": "\033[38;5;45m",
        "reasoning": "\033[38;5;214m",
        "usage": "\033[38;5;78m",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_trace: dict[str, object] | None = None

    def _console_supports_color(self) -> bool:
        if os.environ.get("NO_COLOR"):
            return False
        stream = getattr(sys, "stdout", None)
        return bool(stream and hasattr(stream, "isatty") and stream.isatty())

    def _style_console_text(self, text: str, tone: str, *, bold: bool = False) -> str:
        if not self._console_supports_color():
            return text

        prefix = self._SECTION_COLORS.get(tone, "")
        if bold:
            prefix += self._ANSI_BOLD
        return f"{prefix}{text}{self._ANSI_RESET}"

    def _format_console_section(self, title: str, body: str, tone: str) -> str:
        separator = self._style_console_text("=" * 72, tone)
        heading = self._style_console_text(title, tone, bold=True)
        content = (body or "<empty>").strip() or "<empty>"
        indented = "\n".join(f"  {line}" if line else "" for line in content.splitlines())
        return f"{separator}\n{heading}\n{separator}\n{indented}"

    def _build_trace_payload(self, message, response, *, structured: bool) -> dict[str, object]:
        return {
            "model": str(self.model),
            "structured": structured,
            "content": self._extract_content_text(message) or "",
            "reasoning": self._extract_reasoning_text(message) or "",
            "usage": self._format_usage_trace(response),
        }

    def get_last_trace(self) -> dict[str, object] | None:
        if self.last_trace is None:
            return None
        return dict(self.last_trace)

    def clear_last_trace(self) -> None:
        self.last_trace = None

    def _format_structured_content(self, payload: dict) -> str:
        lines: list[str] = []

        ordered_fields = [
            ("thinking", "Thinking"),
            ("evaluation_previous_goal", "Evaluation"),
            ("memory", "Memory"),
            ("next_goal", "Next Goal"),
            ("current_plan_item", "Current Plan Item"),
        ]
        for key, label in ordered_fields:
            value = payload.get(key)
            text = self._stringify_field(value)
            if text and text.lower() != "null":
                lines.append(f"{label}: {text}")

        plan_update = payload.get("plan_update")
        if isinstance(plan_update, list) and plan_update:
            lines.append("Plan Update:")
            lines.extend(f"- {self._stringify_field(item)}" for item in plan_update if self._stringify_field(item))

        actions = payload.get("action")
        if isinstance(actions, list) and actions:
            lines.append("Actions:")
            for index, action in enumerate(actions, start=1):
                action_text = self._stringify_field(action)
                if action_text:
                    lines.append(f"{index}. {action_text}")

        if not lines:
            return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        return "\n".join(lines)

    def _stringify_field(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if isinstance(item, dict):
                    item_text = item.get("text")
                    if isinstance(item_text, str):
                        parts.append(item_text)
                        continue
                try:
                    parts.append(json.dumps(item, ensure_ascii=False, indent=2, default=str))
                except TypeError:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
        if isinstance(value, dict):
            try:
                return json.dumps(value, ensure_ascii=False, indent=2, default=str)
            except TypeError:
                return str(value)
        return str(value).strip()

    def _extract_reasoning_text(self, message) -> str:
        direct_reasoning = getattr(message, "reasoning_content", None)
        text = self._stringify_field(direct_reasoning)
        if text:
            return text

        model_extra = getattr(message, "model_extra", None)
        if isinstance(model_extra, dict):
            for key in ("reasoning_content", "reasoning", "thinking"):
                text = self._stringify_field(model_extra.get(key))
                if text:
                    return text

        return ""

    def _extract_content_text(self, message) -> str:
        text = self._stringify_field(getattr(message, "content", None))
        if not text:
            return ""

        stripped = self._extract_structured_payload_text(text) or text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                return text
            if isinstance(payload, dict):
                return self._format_structured_content(payload)

        return text

    def _extract_structured_payload_text(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""
        if stripped.startswith("{"):
            return stripped

        start = stripped.find("{")
        if start == -1:
            return ""

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
                continue
            if char == "}":
                depth -= 1
                if depth == 0:
                    return stripped[start : index + 1]

        return ""

    def _format_usage_trace(self, response) -> str:
        usage = getattr(response, "usage", None)
        if usage is None:
            return "<unavailable>"

        lines = [
            f"prompt_tokens: {getattr(usage, 'prompt_tokens', '<unknown>')}",
            f"completion_tokens: {getattr(usage, 'completion_tokens', '<unknown>')}",
            f"total_tokens: {getattr(usage, 'total_tokens', '<unknown>')}",
        ]

        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = getattr(prompt_details, "cached_tokens", None) if prompt_details is not None else None
        if cached_tokens is not None:
            lines.append(f"prompt_cached_tokens: {cached_tokens}")

        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None) if completion_details is not None else None
        if reasoning_tokens is not None:
            lines.append(f"reasoning_tokens: {reasoning_tokens}")

        return "\n".join(lines)

    def _log_console_trace(self, message, response, *, structured: bool) -> None:
        trace_payload = self._build_trace_payload(message, response, structured=structured)
        self.last_trace = trace_payload

        if not getattr(cfg, "SHOW_LLM_CONSOLE_TRACE", False):
            return

        header = self._style_console_text(
            f"[LM Studio Trace] model={trace_payload['model']} structured={trace_payload['structured']}",
            "meta",
            bold=True,
        )
        footer = self._style_console_text("[End LM Studio Trace]", "meta", bold=True)

        print()
        print(header)
        print(self._format_console_section("Content", str(trace_payload["content"]), "content"))
        print()
        print(self._format_console_section("Reasoning Content", str(trace_payload["reasoning"]), "reasoning"))
        print()
        print(self._format_console_section("Usage", str(trace_payload["usage"]), "usage"))
        print(footer)
        print()

    async def ainvoke(self, messages: list[BaseMessage], output_format=None, **kwargs):
        openai_messages = OpenAIMessageSerializer.serialize_messages(messages)

        try:
            model_params: dict[str, object] = {}

            if self.temperature is not None:
                model_params["temperature"] = self.temperature

            if self.frequency_penalty is not None:
                model_params["frequency_penalty"] = self.frequency_penalty

            if self.max_completion_tokens is not None:
                model_params["max_completion_tokens"] = self.max_completion_tokens

            if self.top_p is not None:
                model_params["top_p"] = self.top_p

            if self.seed is not None:
                model_params["seed"] = self.seed

            if self.service_tier is not None:
                model_params["service_tier"] = self.service_tier

            if self.reasoning_models and any(str(candidate).lower() in str(self.model).lower() for candidate in self.reasoning_models):
                model_params["reasoning_effort"] = self.reasoning_effort
                model_params.pop("temperature", None)
                model_params.pop("frequency_penalty", None)

            if output_format is None:
                response = await self.get_client().chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    **model_params,
                )

                choice = response.choices[0] if response.choices else None
                if choice is None:
                    base_url = str(self.base_url) if self.base_url is not None else None
                    hint = f" (base_url={base_url})" if base_url is not None else ""
                    raise ModelProviderError(
                        message=(
                            "Invalid OpenAI chat completion response: missing or empty `choices`."
                            " If you are using a proxy via `base_url`, ensure it implements the OpenAI"
                            " `/v1/chat/completions` schema and returns `choices` as a non-empty list."
                            f"{hint}"
                        ),
                        status_code=502,
                        model=self.name,
                    )

                self._log_console_trace(choice.message, response, structured=False)
                usage = self._get_usage(response)
                return ChatInvokeCompletion(
                    completion=choice.message.content or "",
                    thinking=self._extract_reasoning_text(choice.message) or None,
                    usage=usage,
                    stop_reason=choice.finish_reason,
                )

            response_format: JSONSchema = {
                "name": "agent_output",
                "strict": True,
                "schema": SchemaOptimizer.create_optimized_json_schema(
                    output_format,
                    remove_min_items=self.remove_min_items_from_schema,
                    remove_defaults=self.remove_defaults_from_schema,
                ),
            }

            if self.add_schema_to_system_prompt and openai_messages and openai_messages[0]["role"] == "system":
                schema_text = f"\n<json_schema>\n{response_format}\n</json_schema>"
                if isinstance(openai_messages[0]["content"], str):
                    openai_messages[0]["content"] += schema_text
                elif isinstance(openai_messages[0]["content"], Iterable):
                    openai_messages[0]["content"] = list(openai_messages[0]["content"]) + [
                        ChatCompletionContentPartTextParam(text=schema_text, type="text")
                    ]

            if self.dont_force_structured_output:
                response = await self.get_client().chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    **model_params,
                )
            else:
                response = await self.get_client().chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    response_format=ResponseFormatJSONSchema(json_schema=response_format, type="json_schema"),
                    **model_params,
                )

            choice = response.choices[0] if response.choices else None
            if choice is None:
                base_url = str(self.base_url) if self.base_url is not None else None
                hint = f" (base_url={base_url})" if base_url is not None else ""
                raise ModelProviderError(
                    message=(
                        "Invalid OpenAI chat completion response: missing or empty `choices`."
                        " If you are using a proxy via `base_url`, ensure it implements the OpenAI"
                        " `/v1/chat/completions` schema and returns `choices` as a non-empty list."
                        f"{hint}"
                    ),
                    status_code=502,
                    model=self.name,
                )

            self._log_console_trace(choice.message, response, structured=True)

            if choice.message.content is None:
                raise ModelProviderError(
                    message="Failed to parse structured output from model response",
                    status_code=500,
                    model=self.name,
                )

            usage = self._get_usage(response)
            structured_content = self._extract_structured_payload_text(choice.message.content)
            if not structured_content:
                raise ModelProviderError(
                    message="Failed to locate JSON object in structured model response",
                    status_code=500,
                    model=self.name,
                )

            parsed = output_format.model_validate_json(structured_content)
            return ChatInvokeCompletion(
                completion=parsed,
                thinking=self._extract_reasoning_text(choice.message) or None,
                usage=usage,
                stop_reason=choice.finish_reason,
            )
        except ModelProviderError:
            raise
        except RateLimitError as exc:
            raise ModelRateLimitError(message=exc.message, model=self.name) from exc
        except APIConnectionError as exc:
            raise ModelProviderError(message=str(exc), model=self.name) from exc
        except APIStatusError as exc:
            raise ModelProviderError(message=exc.message, status_code=exc.status_code, model=self.name) from exc
        except Exception as exc:
            raise ModelProviderError(message=str(exc), model=self.name) from exc


def create_lm_studio_llm(model_name: str | None = None, base_url: str | None = None) -> ChatOpenAI:
    return ConsoleTracingChatOpenAI(
        base_url=base_url or LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        model=model_name or MODEL_NAME,
        temperature=0.1,
        dont_force_structured_output=True,
        add_schema_to_system_prompt=True,
    )


def get_llm(model_name: str | None = None) -> ChatOpenAI:
    """Create a ChatOpenAI instance pointed at LM Studio."""
    return create_lm_studio_llm(model_name=model_name)


def get_browser_profile() -> BrowserProfile:
    """Browser configuration for Playwright."""
    return BrowserProfile(
        headless=BROWSER_HEADLESS,
        window_size={"width": BROWSER_WIDTH, "height": BROWSER_HEIGHT},
        keep_alive=True,  # Keep browser open after agent completes so user can verify
    )


class TaskResult:
    """Holds the result of a browser task including optional screenshot."""

    def __init__(
        self,
        text: str,
        screenshot: bytes | None = None,
        tabs: list[dict] | None = None,
        console_trace: dict[str, object] | None = None,
    ):
        self.text = text
        self.screenshot = screenshot  # PNG bytes, or None if unavailable
        self.tabs = tabs or []
        self.console_trace = console_trace or {}


class BrowserAgentSession:
    """Manages a browser session that stays open until the user explicitly closes it."""

    def __init__(self, llm: ChatOpenAI | None = None, max_steps: int | None = None):
        self.llm = llm or get_llm()
        self.max_steps = max_steps if max_steps is not None else MAX_STEPS
        self.browser_session: BrowserSession | None = None
        self.last_result: TaskResult | None = None
        self.tools = build_agent_tools()

    def _get_last_llm_trace(self) -> dict[str, object]:
        if hasattr(self.llm, "get_last_trace"):
            trace = self.llm.get_last_trace()
            if isinstance(trace, dict):
                return trace
        return {}

    async def _emit_step_update(self, step_callback, browser_state, agent_output, step_number: int):
        if step_callback is None:
            return

        next_goal = getattr(agent_output, "next_goal", None)
        actions = getattr(agent_output, "action", None) or []
        action_names = []
        for action in actions:
            if hasattr(action, "model_dump"):
                dumped = action.model_dump(exclude_none=True)
                action_names.extend(dumped.keys())
            elif isinstance(action, dict):
                action_names.extend(action.keys())

        if next_goal:
            message = f"Step {step_number}: {next_goal}"
        elif action_names:
            message = f"Step {step_number}: {' -> '.join(action_names[:3])}"
        else:
            message = f"Step {step_number}: working"

        trace = self._get_last_llm_trace()
        step_payload = {
            "index": step_number,
            "message": message,
            "content": str(trace.get("content", "") or ""),
            "reasoning": str(trace.get("reasoning", "") or ""),
            "usage": str(trace.get("usage", "") or ""),
            "model": str(trace.get("model", "") or ""),
            "structured": bool(trace.get("structured", False)),
        }

        result = step_callback(step_payload)
        if inspect.isawaitable(result):
            await result

    async def get_tabs(self) -> list[dict]:
        if self.browser_session is None:
            return []

        try:
            tabs = await self.browser_session.get_tabs()
        except Exception:
            return []

        return [
            {
                "title": getattr(tab, "title", "") or "Untitled",
                "url": getattr(tab, "url", "") or "",
            }
            for tab in tabs
        ]

    async def run_task(self, task: str, step_callback=None) -> TaskResult:
        """Run a browser task. The browser stays open after completion."""
        # Close any previous session before starting a new one
        await self.close_browser()

        if hasattr(self.llm, "clear_last_trace"):
            self.llm.clear_last_trace()

        self.browser_session = BrowserSession(browser_profile=get_browser_profile())

        step_handler = None
        if step_callback:
            async def step_handler(browser_state, agent_output, step_number: int):
                await self._emit_step_update(step_callback, browser_state, agent_output, step_number)

        agent = Agent(
            task=task,
            llm=self.llm,
            browser_session=self.browser_session,
            tools=self.tools,
            register_new_step_callback=step_handler,
            use_vision=USE_VISION,
            use_judge=False,
            extend_system_message=(
                "IMPORTANT: Before doing anything else on each page, check for and "
                "immediately dismiss any overlays, modals, or banners that block content. "
                "This includes: cookie consent dialogs, privacy notices, newsletter sign-up "
                "popups, interest-based ads notices, GDPR consent forms, age verification "
                "prompts, location permission requests, notification permission requests, "
                "and any other interstitial that has a Close / Accept / Continue / Dismiss / "
                "Got it / OK / No thanks button. Click the least intrusive option "
                "(e.g. 'Close', 'Dismiss', 'No thanks', 'Accept') to get past it, "
                "then proceed with the actual task.\n\n"
                "MULTI-SITE TASKS: Whenever the task requires gathering information from "
                "more than one website (e.g. comparing prices, comparing weather forecasts, "
                "comparing reviews, comparing news coverage, or any other side-by-side "
                "comparison across different sources), open each website in its own new tab "
                "before you start reading any of them. This keeps all pages simultaneously "
                "accessible so you can reference them together when forming your answer. "
                "Do not navigate away from a site you may still need — switch between tabs "
                "instead. After collecting all information, summarise your findings by "
                "explicitly stating which data came from which site.\n\n"
                "FORM CONTROLS: When a required checkbox, radio button, or consent control "
                "does not respond to normal click actions, use set_checkbox_by_label with "
                "the exact visible text next to the checkbox, then verify the state changed "
                "before submitting the form. If a validation message remains visible after "
                "the checkbox state is verified as correct, treat that message as stale UI "
                "until a fresh submit proves otherwise. When login or form text fields misbehave, use "
                "set_input_value_by_label and verify the final value before clicking continue, "
                "sign in, or submit. For buttons and links with stable visible text, "
                "prefer click_element_by_text over fragile click(index) targeting."
            ),
        )

        result = await agent.run(max_steps=self.max_steps)

        # Extract final result text
        result_text = "Task completed."
        if result and hasattr(result, "final_result") and result.final_result:
            result_text = result.final_result()
        elif result and hasattr(result, "history"):
            for entry in reversed(result.history):
                if hasattr(entry, "result") and entry.result:
                    if hasattr(entry.result, "extracted_content") and entry.result.extracted_content:
                        result_text = entry.result.extracted_content
                        break
            else:
                result_text = "Task completed but no text result was extracted."

        # Take a screenshot of the final browser state
        screenshot_bytes = None
        if self.browser_session is not None:
            try:
                screenshot_bytes = await self.browser_session.take_screenshot()
            except Exception:
                pass

        tabs = await self.get_tabs()

        self.last_result = TaskResult(
            text=result_text,
            screenshot=screenshot_bytes,
            tabs=tabs,
            console_trace=self._get_last_llm_trace(),
        )
        return self.last_result

    async def close_browser(self):
        """Close the browser session if one is open."""
        if self.browser_session is not None:
            try:
                await self.browser_session.stop()
            except Exception:
                pass
            self.browser_session = None

    @property
    def is_browser_open(self) -> bool:
        return self.browser_session is not None


async def run_browser_task(task: str, llm: ChatOpenAI | None = None) -> str:
    """Legacy one-shot helper — runs task and closes browser immediately."""
    session = BrowserAgentSession(llm=llm)
    try:
        result = await session.run_task(task)
        return result.text
    finally:
        await session.close_browser()
