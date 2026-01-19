from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def clamp(self, width: int, height: int) -> "BBox":
        x1 = max(0, min(width, self.x1))
        y1 = max(0, min(height, self.y1))
        x2 = max(0, min(width, self.x2))
        y2 = max(0, min(height, self.y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return BBox(x1, y1, x2, y2)


def bbox_from_poly(poly: Iterable[Sequence[float | int]]) -> BBox:
    xs: list[float] = []
    ys: list[float] = []
    for p in poly:
        xs.append(float(p[0]))
        ys.append(float(p[1]))
    return BBox(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def merge_bbox(boxes: Iterable[BBox]) -> BBox:
    boxes = list(boxes)
    if not boxes:
        return BBox(0, 0, 0, 0)
    return BBox(
        x1=min(b.x1 for b in boxes),
        y1=min(b.y1 for b in boxes),
        x2=max(b.x2 for b in boxes),
        y2=max(b.y2 for b in boxes),
    )


def y_overlap_ratio(a: BBox, b: BBox) -> float:
    top = max(a.y1, b.y1)
    bot = min(a.y2, b.y2)
    overlap = max(0, bot - top)
    denom = min(a.h, b.h)
    return overlap / denom if denom > 0 else 0.0
