from __future__ import annotations

import json
import hashlib
import re
import time
from pathlib import Path
import threading
from dataclasses import dataclass
from typing import Any, cast

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)


_JP_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_CJK_IDEOGRAPH_RE = re.compile(r"[\u4e00-\u9fff]")
_HTML_LIKE_RE = re.compile(
    r"^\s*(?:<!doctype\s+html|<html\b|<head\b|<body\b)", re.IGNORECASE)


def looks_like_japanese(text: str) -> bool:
    # 仅靠汉字无法区分中日；按系统提示词要求：只有在“确实不是日文”时才可以不翻译。
    # 这里放宽为：
    # - 有假名：强日文信号
    # - 有 CJK 汉字：也可能是日文（或 OCR 抖动），默认视作“可能需要翻译”
    t = (text or "").strip()
    if not t:
        return False
    return bool(_JP_KANA_RE.search(t) or _CJK_IDEOGRAPH_RE.search(t))


def make_request_id(text: str) -> str:
    """为提示词协议生成稳定 id。

    约束：
    - 同一原文应稳定产生同一 id，便于后续发 context=true
    - id 不需要可逆
    """

    t = (text or "").strip().encode("utf-8", errors="ignore")
    h = hashlib.sha1(t).hexdigest()
    return h[:12]


@dataclass(frozen=True)
class TranslateResult:
    translated: str | None
    contexts: list[str]
    engine: str = "ai"
    warning: str | None = None


class GalgameTranslator:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        system_prompt: str,
        max_history: int = 8,
        min_interval_sec: float = 0.8,
        history_path: str | Path | None = None,
    ):
        api_key = (api_key or "").strip()
        if not api_key:
            raise ValueError("openai.api_key 为空")

        self._client: Any = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=headers,
        )
        self._model = model
        self._system_prompt = system_prompt
        self._max_history = max_history
        self._min_interval_sec = min_interval_sec

        self._history_path = Path(history_path) if history_path else None
        self._history: list[dict[str, str]] = []
        self._pending_context_ids: list[str] = []
        self._history_lock = threading.Lock()

        self._load_history_from_disk()
        self._last_input: str | None = None
        self._last_call_ts: float = 0.0
        self._ai_disabled_until: float = 0.0

    def queue_confirmed_context(self, ids: list[str]) -> None:
        """将用户点击确认的内容（context=true）排队到下一次模型交互中。"""

        ids = [str(x).strip() for x in (ids or []) if str(x).strip()]
        if not ids:
            return
        lock = self._history_lock
        if lock is None:
            self._pending_context_ids.extend(ids)
            self._pending_context_ids = _dedupe_keep_order(
                self._pending_context_ids
            )
            return
        with lock:
            self._pending_context_ids.extend(ids)
            self._pending_context_ids = _dedupe_keep_order(
                self._pending_context_ids
            )

    def send_context_only(self, ids: list[str]) -> None:
        """只发送 context=true 行（不请求翻译），用于满足“每次流程开始前先发汇总”的需求。"""

        ids = [str(x).strip() for x in (ids or []) if str(x).strip()]
        if not ids:
            return

        user_content = "\n".join(
            json.dumps(
                {"id": cid, "context": True},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for cid in ids
        )
        self._call_ai(user_content)

    def translate(self, jp_text: str) -> TranslateResult:
        jp_text = (jp_text or "").strip()
        if not jp_text:
            return TranslateResult(translated=None, contexts=[])
        if not looks_like_japanese(jp_text):
            return TranslateResult(translated=None, contexts=[])

        if self._last_input == jp_text:
            return TranslateResult(translated=None, contexts=[])

        now = time.time()
        since = now - self._last_call_ts
        if since < self._min_interval_sec:
            # 节流：避免 OCR 抖动导致短时间连续请求
            return TranslateResult(translated=None, contexts=[])

        self._last_call_ts = now
        self._last_input = jp_text

        if time.time() < self._ai_disabled_until:
            return TranslateResult(
                translated=None,
                contexts=[],
                engine="ai",
                warning="AI 翻译暂时熔断（上一轮失败），稍后会自动重试。",
            )

        try:
            req_id = make_request_id(jp_text)
            req_line = json.dumps(
                {"id": req_id, "text": jp_text},
                ensure_ascii=False,
                separators=(",", ":"),
            )

            context_lines: list[str] = []
            lock = self._history_lock
            if lock is None:
                pending = list(self._pending_context_ids)
                self._pending_context_ids = []
            else:
                with lock:
                    pending = list(self._pending_context_ids)
                    self._pending_context_ids = []

            if pending:
                context_lines = [
                    json.dumps(
                        {"id": cid, "context": True},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    for cid in pending
                ]

            user_content = "\n".join([*context_lines, req_line])
            text_out = self._call_ai(user_content)
            if not text_out:
                # AI 无输出：按失败处理，允许走 fallback
                raise RuntimeError("AI 无输出")

            if _looks_like_html(text_out):
                base_url = getattr(
                    getattr(self._client, "_client", None), "base_url", None)
                snippet = text_out[:240].replace("\n", " ")
                raise RuntimeError(
                    "上游返回了 HTML（疑似被 Cloudflare/站点安全策略拦截，"
                    "或 base_url 配置成了网页地址而非 API）。"
                    f" 请检查 config.toml 的 [openai].base_url。"
                    f"base_url={base_url!r} snippet={snippet!r}"
                )

            items = _parse_json_lines(text_out)
        except (
            APIConnectionError,
            APITimeoutError,
            APIStatusError,
            AuthenticationError,
            PermissionDeniedError,
            RateLimitError,
            BadRequestError,
            NotFoundError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as e:
            # AI 请求失败：短暂熔断，避免高频重复触发风控/拦截
            self._ai_disabled_until = time.time() + 20.0
            return TranslateResult(
                translated=None,
                contexts=[],
                engine="ai",
                warning=f"AI 翻译失败：{type(e).__name__}",
            )

        translated: str | None = None
        contexts: list[str] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            if str(it.get("id", "")).strip() != req_id:
                continue
            t = str(it.get("type", "")).strip()
            content = str(it.get("text", "")).strip()
            if not content:
                continue
            if t == "translate":
                translated = content
            elif t == "context":
                contexts.append(content)

        # 如果 AI 没给任何可用内容，返回空结果即可（不做 fallback）

        return TranslateResult(
            translated=translated,
            contexts=contexts,
            engine="ai",
            warning=None,
        )

    def _call_ai(self, user_content: str) -> str:
        if self._client is None:
            raise RuntimeError("AI client 未初始化")

        # OpenAI SDK 的 typing 比较严格；这里用 cast(Any) 让静态检查通过。
        messages: Any = [
            {"role": "system", "content": self._system_prompt},
            *self._history[-(self._max_history * 2):],
            {"role": "user", "content": user_content},
        ]

        # 注意：为了兼容更多 openai-compatible 服务，这里使用 chat.completions。
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=cast(Any, messages),
        )
        content = resp.choices[0].message.content if resp.choices else None
        text_out = (content or "").strip()
        if _looks_like_html(text_out):
            base_url = getattr(
                getattr(self._client, "_client", None), "base_url", None)
            snippet = text_out[:240].replace("\n", " ")
            raise RuntimeError(
                "上游返回了 HTML（疑似被 Cloudflare/站点安全策略拦截，"
                "或 base_url 配置成了网页地址而非 API）。"
                f" 请检查 config.toml 的 [openai].base_url。"
                f"base_url={base_url!r} snippet={snippet!r}"
            )

        # 始终保存与大模型交互的记录（即使输出为空也保存 user）。
        self._append_history("user", user_content)
        if text_out:
            self._append_history("assistant", text_out)
        return text_out

    def _append_history(self, role: str, content: str) -> None:
        rec = {"role": str(role), "content": str(content)}
        lock = self._history_lock
        if lock is None:
            self._history.append(rec)
            self._history = self._history[-(self._max_history * 2):]
            self._append_history_to_disk(rec)
            return
        with lock:
            self._history.append(rec)
            self._history = self._history[-(self._max_history * 2):]
            self._append_history_to_disk(rec)

    def _append_history_to_disk(self, rec: dict[str, str]) -> None:
        if self._history_path is None:
            return
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(rec, ensure_ascii=False)
            self._history_path.open("a", encoding="utf-8").write(line + "\n")
        except OSError:
            # 持久化失败不应阻塞主流程
            pass

    def _load_history_from_disk(self) -> None:
        if self._history_path is None:
            return
        try:
            if not self._history_path.exists():
                return
            lines = self._history_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return

        loaded: list[dict[str, str]] = []
        for ln in lines[-200:]:
            ln = (ln or "").strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            role = str(obj.get("role", "")).strip()
            content = str(obj.get("content", ""))
            if role not in {"user", "assistant", "system"}:
                continue
            if not content:
                continue
            loaded.append({"role": role, "content": content})

        if loaded:
            self._history = loaded[-(self._max_history * 2):]


def _parse_json_lines(s: str) -> list[dict[str, Any]]:
    """按系统提示词的“每行一个 JSON”协议解析模型输出。"""

    s = (s or "").strip()
    if not s:
        return []

    # 处理 ```json ... ``` 或 ``` ... ``` 代码块
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json\n", "", 1).strip()

    out: list[dict[str, Any]] = []
    for ln in s.splitlines():
        ln = (ln or "").strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            # 允许模型把多个 JSON 拼成数组/对象的情况
            try:
                obj = json.loads(_extract_jsonish(ln))
            except json.JSONDecodeError:
                continue
        if isinstance(obj, dict):
            out.append(obj)
        elif isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict):
                    out.append(it)
    return out


def _extract_jsonish(s: str) -> str:
    s = (s or "").strip()
    l1 = s.find("{")
    r1 = s.rfind("}")
    if 0 <= l1 < r1:
        return s[l1: r1 + 1]
    l2 = s.find("[")
    r2 = s.rfind("]")
    if 0 <= l2 < r2:
        return s[l2: r2 + 1]
    return s


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _looks_like_html(s: str) -> bool:
    return bool(_HTML_LIKE_RE.match(s))
