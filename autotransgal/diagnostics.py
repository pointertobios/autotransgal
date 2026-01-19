from __future__ import annotations

import json
import socket
from urllib.parse import urlparse

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from autotransgal.config import OpenAIConfig


def diagnose_openai_compatible(cfg: OpenAIConfig) -> str:
    """探测 openai-compatible 服务是否可用，并输出可复制粘贴的诊断文本。"""

    base_url = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")

    parsed = urlparse(base_url)
    host = parsed.hostname
    ip = None
    if host:
        try:
            ip = socket.gethostbyname(host)
        except OSError:
            ip = None

    lines: list[str] = []
    lines.append("== autotransgal openai-compatible 诊断 ==")
    lines.append(f"base_url: {base_url}")
    if host:
        lines.append(f"host: {host} ip: {ip or 'N/A'}")
    lines.append(f"model: {cfg.model}")
    if cfg.headers:
        headers_json = json.dumps(cfg.headers, ensure_ascii=False)
        lines.append(f"extra_headers: {headers_json}")

    client = OpenAI(
        api_key=cfg.api_key,
        base_url=base_url,
        default_headers=cfg.headers or None,
        timeout=10.0,
        max_retries=0,
    )

    try:
        models = client.models.list()
    except AuthenticationError as e:
        lines.append("\n-- 结论 --")
        lines.append("鉴权失败：请检查 api_key 是否正确/是否有权限访问该服务。")
        lines.append(f"{type(e).__name__}: {e}")
        return "\n".join(lines)
    except PermissionDeniedError as e:
        lines.append("\n-- 结论 --")
        lines.append("无权限：请检查服务侧的权限策略/模型权限。")
        lines.append(f"{type(e).__name__}: {e}")
        return "\n".join(lines)
    except RateLimitError as e:
        lines.append("\n-- 结论 --")
        lines.append("触发限流：请稍后重试或提升额度/并发限制。")
        lines.append(f"{type(e).__name__}: {e}")
        return "\n".join(lines)
    except APITimeoutError as e:
        lines.append("\n-- 结论 --")
        lines.append("请求超时：请检查网络连通性，或调大超时。")
        lines.append(f"{type(e).__name__}: {e}")
        return "\n".join(lines)
    except APIConnectionError as e:
        lines.append("\n-- 结论 --")
        lines.append("连接失败：请检查 base_url/DNS/代理/证书/网络。")
        lines.append(f"{type(e).__name__}: {e}")
        if e.__cause__ is not None:
            lines.append(f"cause: {type(e.__cause__).__name__}: {e.__cause__}")
        return "\n".join(lines)
    except APIStatusError as e:
        lines.append("\n-- 响应 --")
        lines.append(f"status: {e.status_code}")
        body = getattr(e, "body", None)
        if isinstance(body, str):
            snippet = body[:400].replace("\n", " ").strip()
            lines.append(f"body_snippet: {snippet!r}")
            low = snippet.lower()
            if "<html" in low or "<!doctype html" in low:
                lines.append("\n-- 结论 --")
                lines.append(
                    "返回了 HTML（常见于 Cloudflare/WAF：Please enable cookies / "
                    "Sorry you have been blocked）。"
                )
                lines.append(
                    "这不是可供 SDK 调用的 API 响应。请让服务方提供不需要 Cookie/JS 的 API 域名，"
                    "或将 /v1/* 从网页防护中放行。"
                )
            else:
                lines.append("\n-- 结论 --")
                lines.append("服务返回非 2xx：可能是鉴权/路径/权限/限流问题。")
        else:
            lines.append(f"body: {body!r}")
            lines.append("\n-- 结论 --")
            lines.append("服务返回非 2xx：可能是鉴权/路径/权限/限流问题。")
        return "\n".join(lines)

    model_ids = [m.id for m in (models.data or []) if getattr(m, "id", None)]
    lines.append("\n-- 响应 --")
    lines.append(f"models: {len(model_ids)}")
    if model_ids:
        shown = ", ".join(model_ids[:8])
        lines.append(f"sample: {shown}{' ...' if len(model_ids) > 8 else ''}")
    lines.append("\n-- 结论 --")
    lines.append("看起来像正常 API（通过 OpenAI SDK 成功列出 models）。")

    return "\n".join(lines)
