"""已废弃：Google 翻译 fallback。

用户已决定完全抛弃该兜底逻辑；此文件仅保留同名符号以避免旧引用。
"""

from __future__ import annotations


class GoogleTranslateError(RuntimeError):
    pass


def google_translate_ja_to_zh_cn(
    text: str,
    *,
    timeout_sec: float = 8.0,
) -> str:
    raise GoogleTranslateError(
        "Google 翻译 fallback 已被移除；请使用 OpenAI Python SDK 的 AI 翻译。"
    )
