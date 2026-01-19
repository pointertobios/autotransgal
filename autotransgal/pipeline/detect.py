from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from autotransgal.util.geometry import BBox, bbox_from_poly


@dataclass(frozen=True)
class DetectedRegion:
    bbox: BBox
    score: float


class TextRegionDetector:
    def __init__(self):
        # RapidOCR 默认会带 det/rec/cls 模型。这里我们只用 det。
        self._ocr = RapidOCR()

    def detect(self, image: Image.Image) -> list[DetectedRegion]:
        arr = np.asarray(image)
        result, _elapsed = self._ocr(
            arr, use_det=True, use_cls=False, use_rec=False)
        if not result:
            return []

        regions: list[DetectedRegion] = []
        for item in result:
            # RapidOCR 的返回结构在不同模式下不一样：
            # - det-only: result = [poly, poly, ...]，
            #   每个 poly = [[x,y], [x,y], [x,y], [x,y]]（没有 score）
            # - det+rec: item 通常是 [poly, text, score]
            if not item:
                continue

            poly = None
            score = 1.0

            if _looks_like_poly(item):
                poly = item
                score = 1.0
            elif (
                isinstance(item, (list, tuple))
                and item
                and _looks_like_poly(item[0])
            ):
                poly = item[0]
                score = _extract_score(item[2]) if len(
                    item) >= 3 else _extract_score(item[-1])
            else:
                continue
            try:
                bbox = bbox_from_poly(poly)
            except (TypeError, ValueError, IndexError):
                continue
            regions.append(DetectedRegion(bbox=bbox, score=score))

        return regions


def _looks_like_poly(obj: object) -> bool:
    if not isinstance(obj, (list, tuple)):
        return False
    if len(obj) < 4:
        return False
    for p in obj:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False
    return True


def _extract_score(x: object) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return 1.0
    return 1.0


def filter_regions(
    regions: Iterable[DetectedRegion], *, min_area: int, max_regions: int
) -> list[DetectedRegion]:
    kept = [r for r in regions if r.bbox.area >= min_area]
    kept.sort(key=lambda r: (r.bbox.y1, r.bbox.x1))
    return kept[: max(0, int(max_regions))]
