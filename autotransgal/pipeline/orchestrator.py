from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field

from autotransgal.config import AppConfig, SYSTEM_PROMPT
from autotransgal.pipeline.capture import Frame, ScreenCapturer
from autotransgal.pipeline.detect import TextRegionDetector, filter_regions
from autotransgal.pipeline.merge import merge_and_sort
from autotransgal.pipeline.recognize import TextRecognizer
from autotransgal.pipeline.translate import GalgameTranslator
from autotransgal.util.geometry import BBox


@dataclass(frozen=True)
class RegionOutput:
    bbox: BBox
    jp_text: str
    translated: str | None
    engine: str
    warning: str | None


@dataclass(frozen=True)
class PipelineOutput:
    regions: list[RegionOutput]
    # 调试/可视化：合并前/合并后区域框（屏幕绝对坐标）
    debug_boxes_pre: list[BBox] = field(default_factory=list)
    debug_boxes_merged: list[BBox] = field(default_factory=list)


class Pipeline:
    def __init__(self, cfg: AppConfig):
        self._cfg = cfg
        self._capturer = ScreenCapturer(
            monitor=cfg.capture.monitor, backend=cfg.capture.backend)
        self._core = PipelineCore(cfg)

    def tick(self) -> PipelineOutput:
        frame = self._capturer.grab()
        return self._core.process_frame(frame)


class PipelineCore:
    def __init__(self, cfg: AppConfig):
        self._cfg = cfg
        self._detector = TextRegionDetector()
        self._recognizer = TextRecognizer(force_cpu=False)
        self._translator = GalgameTranslator(  # type: ignore[call-arg]
            api_key=cfg.openai.api_key,
            model=cfg.openai.model,
            base_url=cfg.openai.base_url,
            headers=cfg.openai.headers,
            system_prompt=SYSTEM_PROMPT,
            min_interval_sec=cfg.translate.min_interval_sec,
            history_path=(cfg.config_path.parent / "ai_history.jsonl"),
        )

        self._last_emit_text: str | None = None
        self._last_emit_ts: float = 0.0
        self._tr_cache: dict[str, tuple[float, object]] = {}

    def queue_confirmed_context(self, ids: list[str]) -> None:
        """将用户确认的内容 id 排队到下一次翻译请求中（context=true）。"""

        self._translator.queue_confirmed_context(ids)

    def send_context_only(self, ids: list[str]) -> None:
        """可选：只发送 context=true 行，不请求翻译。"""

        self._translator.send_context_only(ids)

    def recognize_regions(
        self,
        frame: Frame,
        roi: BBox | None = None,
    ) -> list[RegionOutput]:
        """只做检测+OCR，返回每个区域的识别结果（不等待翻译）。"""

        roi_rel: BBox | None = None
        if roi is not None:
            roi_rel = BBox(
                int(roi.x1) - int(frame.left),
                int(roi.y1) - int(frame.top),
                int(roi.x2) - int(frame.left),
                int(roi.y2) - int(frame.top),
            ).clamp(frame.width, frame.height)
            if roi_rel.x2 <= roi_rel.x1 or roi_rel.y2 <= roi_rel.y1:
                return []

        if roi_rel is None:
            det_img = frame.image
            offset_x = 0
            offset_y = 0
        else:
            det_img = frame.image.crop(
                (roi_rel.x1, roi_rel.y1, roi_rel.x2, roi_rel.y2)
            )
            offset_x = int(roi_rel.x1)
            offset_y = int(roi_rel.y1)

        regions = self._detector.detect(det_img)
        regions = filter_regions(
            regions,
            min_area=self._cfg.ocr.det_min_box_area,
            max_regions=self._cfg.ocr.max_regions,
        )

        out: list[RegionOutput] = []

        # ROI 内检测不到框时，fallback：直接对 ROI 整块做一次识别
        if not regions and roi_rel is not None:
            bbox = roi_rel
            jp = (self._recognizer.recognize(frame.image, bbox) or "").strip()
            if not jp:
                return []

            abs_bbox = BBox(
                bbox.x1 + int(frame.left),
                bbox.y1 + int(frame.top),
                bbox.x2 + int(frame.left),
                bbox.y2 + int(frame.top),
            )

            return [
                RegionOutput(
                    bbox=abs_bbox,
                    jp_text=jp,
                    translated=None,
                    engine="ai",
                    warning=None,
                )
            ]

        for r in regions:
            # r.bbox 是 det_img 的局部坐标；转回 frame.image 的坐标
            bbox = BBox(
                int(r.bbox.x1) + offset_x,
                int(r.bbox.y1) + offset_y,
                int(r.bbox.x2) + offset_x,
                int(r.bbox.y2) + offset_y,
            ).clamp(frame.width, frame.height)

            jp = (self._recognizer.recognize(frame.image, bbox) or "").strip()
            if not jp:
                continue

            abs_bbox = BBox(
                bbox.x1 + int(frame.left),
                bbox.y1 + int(frame.top),
                bbox.x2 + int(frame.left),
                bbox.y2 + int(frame.top),
            )

            out.append(
                RegionOutput(
                    bbox=abs_bbox,
                    jp_text=jp,
                    translated=None,
                    engine="ai",
                    warning=None,
                )
            )

        return out

    def translate_regions(
        self,
        regions: list[RegionOutput],
        *,
        cancel_event: threading.Event | None = None,
    ) -> list[RegionOutput]:
        """为已识别的区域补齐翻译结果（可能命中缓存）。"""

        now = time.time()

        if len(self._tr_cache) > 192:
            items = sorted(self._tr_cache.items(), key=lambda kv: kv[1][0])
            for k, _ in items[:64]:
                self._tr_cache.pop(k, None)

        per_batch_seen: dict[str, object] = {}
        out: list[RegionOutput] = []

        for idx, r in enumerate(regions):
            if cancel_event is not None and cancel_event.is_set():
                # 取消：对剩余区域保持未翻译状态，保证输出长度稳定
                out.extend(regions[idx:])
                break

            jp = (r.jp_text or "").strip()
            if not jp:
                out.append(r)
                continue

            tr_obj = None
            if jp in per_batch_seen:
                tr_obj = per_batch_seen[jp]
            else:
                cached = self._tr_cache.get(jp)
                if cached and (now - float(cached[0])) < 30.0:
                    tr_obj = cached[1]
                else:
                    if cancel_event is not None and cancel_event.is_set():
                        out.extend(regions[idx:])
                        break
                    tr_obj = self._translator.translate(jp)
                    self._tr_cache[jp] = (now, tr_obj)
                per_batch_seen[jp] = tr_obj

            translated = getattr(tr_obj, "translated", None)
            engine = getattr(tr_obj, "engine", "ai")
            warning = getattr(tr_obj, "warning", None)

            out.append(
                RegionOutput(
                    bbox=r.bbox,
                    jp_text=jp,
                    translated=translated,
                    engine=engine,
                    warning=warning,
                )
            )

        return out

    def process_frame(
        self,
        frame: Frame,
        roi: BBox | None = None,
    ) -> PipelineOutput:
        recognized_raw = self.recognize_regions(frame, roi=roi)
        merged = merge_and_sort(
            [(r.bbox, r.jp_text) for r in recognized_raw],
            merge_gap_in_line_ratio=float(
                self._cfg.ocr.merge_gap_in_line_ratio
            ),
            merge_gap_between_lines_ratio=float(
                self._cfg.ocr.merge_gap_between_lines_ratio
            ),
        )

        recognized_regions = [
            RegionOutput(
                bbox=m.bbox,
                jp_text=m.text,
                translated=None,
                engine="ai",
                warning=None,
            )
            for m in merged
            if (m.text or "").strip()
        ]

        out_regions = self.translate_regions(recognized_regions)
        main_text = _pick_main_text(merged) or ""
        now = time.time()

        # 节流：如果主文本的译文连续相同且间隔很短，就不重复更新（但仍返回 regions）
        if main_text:
            cached_main = self._tr_cache.get(main_text)
            main_tr = cached_main[1] if cached_main else None
            main_out = (
                getattr(main_tr, "translated", None) if main_tr else None
            )
            if (
                main_out
                and main_out == self._last_emit_text
                and (now - self._last_emit_ts) < 0.6
            ):
                return PipelineOutput(regions=out_regions)

            if main_out:
                self._last_emit_text = main_out
                self._last_emit_ts = now

        return PipelineOutput(
            regions=out_regions,
            debug_boxes_pre=[r.bbox for r in recognized_raw],
            debug_boxes_merged=[m.bbox for m in merged],
        )


def _pick_main_text(merged: list) -> str | None:
    if not merged:
        return None

    # galgame 通常主文本在屏幕下方：优先选择 y 最大的那一行里最长的文本
    # merged 已经按 y/x 排序，这里先找最底部区域
    max_y = max(m.bbox.y2 for m in merged)
    bottom = [m for m in merged if (max_y - m.bbox.y2) <= 20]
    if not bottom:
        bottom = merged

    bottom.sort(key=lambda m: len(m.text), reverse=True)
    candidate = bottom[0].text.strip() if bottom else ""
    return candidate or None
