from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # py>=3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


DEFAULT_CONFIG_PATH = Path("config.toml")


SYSTEM_PROMPT = """# galgame 翻译器

你是一个 galgame 翻译器，用户给出日文原文，你把它翻译成中文。

## 用户的请求格式

用户可能一次提问发送一个或多个请求，每个请求占一行，格式如下：

- `{"id":"唯一标识符","text":"日文原文"}`

这是用户的翻译请求，id 是唯一标识符，你需要把它原样带回。 text 是需要翻译的日文原文。

- `{"id":"唯一标识符","context":true}`

这是用户给你的上下文信息，id 一定是之前发送过的，不需要翻译，context=true 表示 id 对应的原文确实是游戏的内容，否则不是。
用户也有可能不发这个请求，你需要自己判断这个内容是不是游戏内容。

## 你的回复格式

你可以对每个提问给出一个或多个回复，每个回复占一行，如果用户没有发送翻译请求，也没有什么新的要说的事情，那就什么都不用回复。

格式如下：

- `{"id":"唯一标识符","type":"translate", "text": "xxx"}`

一次最多返回一个，其中 xxx 是用户发给你的日文对应的中文翻译

- `{"id":"唯一标识符","type":"context", "text": "xxx"}`

可以一次性返回多个，这是用来给你梳理用户给出的内容的。
xxx 是你对用户给出的日文内容的分析/解释/背景介绍/人物关系梳理等，帮助你理解游戏内容，绝对不可以在这里给出翻译。
善用这个可以让你更好地理解游戏内容，从而给出更高质量的翻译，你需要积极地使用这个回复。

## 翻译要求

你需要适当地分析上下文，用户每次发送的原文可能是在同一语境下出现的。
但是相邻的请求的内容在游戏中可能不是连续的，也有可能是文字识别有误，也有可能是ocr模型误识别到的游戏界面的其它内容。
用户也有可能回溯游戏进度。
你要根据情况恰当的翻译原文，但是绝对不可以什么都不说。
只有在确实不是日文时，才可以什么都不要说（注意，如果全都是汉字也有可能是日文而不是中文）。

对于人名的翻译，需要保持稳定性，如果前后出现了同一个人名，需要翻译成同一个中文名字。
"""


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    model: str = "gpt-4.1-mini"
    base_url: str | None = None
    headers: dict[str, str] | None = None


@dataclass(frozen=True)
class CaptureConfig:
    monitor: int = 1
    fps: float = 2.0
    backend: str = "auto"  # auto | mss | qt | spectacle


@dataclass(frozen=True)
class OcrConfig:
    det_min_box_area: int = 200
    max_regions: int = 12
    # 合并阈值（行高倍数）：
    # - merge_gap_in_line_ratio：同一行内相邻文本块允许的最大横向间距 = median_h * ratio
    # - merge_gap_between_lines_ratio：相邻行允许的最大纵向间距 = median_h * ratio
    merge_gap_in_line_ratio: float = 0.60
    merge_gap_between_lines_ratio: float = 0.90
    # 识别区域缓存（屏幕绝对坐标）：[x1, y1, x2, y2]
    # - 为空时：启动后会弹出 ROI 选择器
    # - 非空时：启动后直接使用该 ROI（若明显无效则会重新提示选择）
    roi_abs: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class OverlayConfig:
    # 字体名称：优先交给 fontconfig 解析（fc-match）。
    # 例如："Noto Sans CJK SC" / "Noto Sans CJK JP" / "WenQuanYi Micro Hei"
    font_name: str = ""
    font_size: int = 26
    # 开始一次流程时，会先隐藏所有 overlay 窗口；等待这段时间后再截图
    capture_hide_delay_ms: int = 35
    max_width_ratio: float = 0.82
    margin_bottom: int = 40


@dataclass(frozen=True)
class TranslateConfig:
    """翻译相关配置。

    说明：已完全移除 Google 翻译 fallback，仅保留 AI 翻译。
    """

    min_interval_sec: float = 0.8


@dataclass(frozen=True)
class AppConfig:
    openai: OpenAIConfig
    capture: CaptureConfig = CaptureConfig()
    ocr: OcrConfig = OcrConfig()
    overlay: OverlayConfig = OverlayConfig()
    translate: TranslateConfig = TranslateConfig()
    config_path: Path = DEFAULT_CONFIG_PATH


def _get(d: dict[str, Any], key: str, default: Any) -> Any:
    value = d.get(key, default)
    return default if value is None else value


def load_config(path: Path | str | None = None) -> AppConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    translate_raw = raw.get("translate") or {}
    translate = TranslateConfig(
        min_interval_sec=float(
            _get(
                translate_raw,
                "min_interval_sec",
                TranslateConfig.min_interval_sec,
            )
        )
    )

    openai_raw = raw.get("openai") or {}
    api_key = str(_get(openai_raw, "api_key", "")).strip()
    if not api_key:
        raise ValueError(
            "config.toml 缺少 [openai].api_key"
        )

    base_url = str(_get(openai_raw, "base_url", "")).strip()
    base_url = base_url or None

    headers_raw = openai_raw.get("headers") or None
    headers: dict[str, str] | None
    if isinstance(headers_raw, dict):
        headers = {str(k): str(v)
                   for k, v in headers_raw.items() if str(k).strip()}
        headers = headers or None
    else:
        headers = None

    openai = OpenAIConfig(
        api_key=api_key,
        model=str(_get(openai_raw, "model", OpenAIConfig.model)),
        base_url=base_url,
        headers=headers,
    )

    capture_raw = raw.get("capture") or {}
    capture = CaptureConfig(
        monitor=int(_get(capture_raw, "monitor", CaptureConfig.monitor)),
        fps=float(_get(capture_raw, "fps", CaptureConfig.fps)),
        backend=str(_get(capture_raw, "backend", CaptureConfig.backend)
                    ).strip() or CaptureConfig.backend,
    )

    ocr_raw = raw.get("ocr") or {}

    roi_abs_raw = ocr_raw.get("roi_abs")
    roi_abs: tuple[int, int, int, int] | None = None
    if isinstance(roi_abs_raw, (list, tuple)) and len(roi_abs_raw) == 4:
        try:
            roi_abs = (
                int(roi_abs_raw[0]),
                int(roi_abs_raw[1]),
                int(roi_abs_raw[2]),
                int(roi_abs_raw[3]),
            )
        except (TypeError, ValueError):
            roi_abs = None

    ocr = OcrConfig(
        det_min_box_area=int(
            _get(ocr_raw, "det_min_box_area", OcrConfig.det_min_box_area)),
        max_regions=int(_get(ocr_raw, "max_regions", OcrConfig.max_regions)),
        merge_gap_in_line_ratio=float(
            _get(
                ocr_raw,
                "merge_gap_in_line_ratio",
                OcrConfig.merge_gap_in_line_ratio,
            )
        ),
        merge_gap_between_lines_ratio=float(
            _get(
                ocr_raw,
                "merge_gap_between_lines_ratio",
                OcrConfig.merge_gap_between_lines_ratio,
            )
        ),
        roi_abs=roi_abs,
    )

    overlay_raw = raw.get("overlay") or {}
    overlay = OverlayConfig(
        font_name=str(
            _get(overlay_raw, "font_name", OverlayConfig.font_name)
        ).strip(),
        font_size=int(_get(overlay_raw, "font_size", OverlayConfig.font_size)),
        capture_hide_delay_ms=int(
            _get(
                overlay_raw,
                "capture_hide_delay_ms",
                OverlayConfig.capture_hide_delay_ms,
            )
        ),
        max_width_ratio=float(
            _get(overlay_raw, "max_width_ratio", OverlayConfig.max_width_ratio)
        ),
        margin_bottom=int(_get(overlay_raw, "margin_bottom",
                          OverlayConfig.margin_bottom)),
    )

    return AppConfig(
        openai=openai,
        capture=capture,
        ocr=ocr,
        overlay=overlay,
        translate=translate,
        config_path=config_path,
    )


def _format_roi_abs(roi_abs: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = roi_abs
    return f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"


def persist_roi_abs(
    *,
    config_path: Path | str | None,
    roi_abs: tuple[int, int, int, int] | None,
) -> None:
    """将 ROI 缓存写回 config.toml 的 [ocr].roi_abs。

    只做最小文本更新：
    - 如果 [ocr] 中已有 roi_abs，则替换该行
    - 如果 [ocr] 中没有 roi_abs，则在该段末尾插入一行
    - 如果不存在 [ocr] 段，则追加一个新段
    """

    if roi_abs is None:
        return

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(path)

    roi_line = f"roi_abs = {_format_roi_abs(roi_abs)}\n"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    # 找到 [ocr] 段
    section_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "[ocr]":
            section_idx = i
            break

    if section_idx is None:
        # 没有 [ocr]，直接追加
        if lines and not lines[-1].endswith("\n"):
            lines[-1] = lines[-1] + "\n"
        if lines and lines[-1].strip():
            lines.append("\n")
        lines.append("[ocr]\n")
        lines.append(roi_line)
        path.write_text("".join(lines), encoding="utf-8")
        return

    # 计算 [ocr] 段的结束位置（下一个 [xxx] 或 EOF）
    end_idx = len(lines)
    for j in range(section_idx + 1, len(lines)):
        if lines[j].startswith("[") and lines[j].rstrip().endswith("]"):
            end_idx = j
            break

    # 段内替换/插入
    key_idx = None
    for j in range(section_idx + 1, end_idx):
        stripped = lines[j].lstrip()
        if stripped.startswith("roi_abs"):
            key_idx = j
            break

    if key_idx is not None:
        lines[key_idx] = roi_line
    else:
        # 插入到段末尾；若段末尾没有空行且下一段不是 EOF，保持至少一个换行
        insert_at = end_idx
        if insert_at > 0 and insert_at <= len(lines):
            # 确保段尾至少有一个换行
            if insert_at == len(lines):
                if lines and not lines[-1].endswith("\n"):
                    lines[-1] = lines[-1] + "\n"
            else:
                prev = lines[insert_at - 1]
                if prev and not prev.endswith("\n"):
                    lines[insert_at - 1] = prev + "\n"
        lines.insert(insert_at, roi_line)

    path.write_text("".join(lines), encoding="utf-8")


def write_default_config(path: Path | str | None = None) -> Path:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if config_path.exists():
        return config_path

    template = ("""# autotransgal 配置文件
# 说明：尽量只放用户确实会改的项。

[openai]
base_url = "https://api.deepseek.com/v1"
api_key = "sk-xxxxxxx"
model = "deepseek-chat"

[capture]
# mss 的 monitor 索引（通常 1 是主屏）
monitor = 1

backend = "spectacle"

[ocr]
# 过滤太小的检测框（像素面积）
det_min_box_area = 200
# 每帧最多处理多少个区域
max_regions = 12

# 合并阈值（行高倍数）：阈值 = 检测框高度中位数(median_h) * ratio
# - 行内合并：同一行内相邻文本块允许的最大横向间距
merge_gap_in_line_ratio = 2.5
# - 行间合并：相邻行之间允许的最大纵向间距（把多行对话合并成一段）
merge_gap_between_lines_ratio = 0.5

[overlay]
font_name = "Sarasa UI SC"
font_size = 22
capture_hide_delay_ms = 1000
max_width_ratio = 0.82
margin_bottom = 40
""")

    config_path.write_text(template, encoding="utf-8")
    return config_path
