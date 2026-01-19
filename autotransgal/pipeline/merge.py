from __future__ import annotations

from dataclasses import dataclass

from autotransgal.util.geometry import BBox, merge_bbox, y_overlap_ratio


@dataclass(frozen=True)
class MergedRegion:
    bbox: BBox
    text: str


def merge_and_sort(
    recognized: list[tuple[BBox, str]],
    *,
    merge_gap_in_line_ratio: float = 0.60,
    merge_gap_between_lines_ratio: float = 0.90,
) -> list[MergedRegion]:
    """对识别区域做基础合并与阅读顺序排序。

    目标是适配 galgame 常见的“底部横向文本框”，同时对少量碎片框做合并。
    算法偏保守：先按行聚类（y 重叠），再在行内按 x 排序并按间距合并。
    """

    items = [(b, t.strip()) for (b, t) in recognized if t and t.strip()]
    if not items:
        return []

    # 估计字号尺度：用中位数高度来推阈值
    heights = sorted(b.h for b, _ in items)
    median_h = heights[len(heights) // 2] if heights else 20
    same_line_overlap = 0.55
    in_line_ratio = max(0.0, float(merge_gap_in_line_ratio))
    between_line_ratio = max(0.0, float(merge_gap_between_lines_ratio))
    in_line_gap = max(0, int(median_h * in_line_ratio))
    between_line_gap = max(0, int(median_h * between_line_ratio))

    # 1) 行聚类
    lines: list[list[tuple[BBox, str]]] = []
    for box, text in sorted(items, key=lambda it: (it[0].cy, it[0].cx)):
        placed = False
        for line in lines:
            ref_box = line[0][0]
            if y_overlap_ratio(ref_box, box) >= same_line_overlap:
                line.append((box, text))
                placed = True
                break
        if not placed:
            lines.append([(box, text)])

    # 2) 行内排序 + 合并
    merged: list[MergedRegion] = []
    for line in sorted(lines, key=lambda ln: min(b.y1 for b, _ in ln)):
        line_sorted = sorted(line, key=lambda it: it[0].x1)

        cur_boxes: list[BBox] = []
        cur_texts: list[str] = []
        last_box: BBox | None = None
        for box, text in line_sorted:
            if last_box is None:
                cur_boxes = [box]
                cur_texts = [text]
                last_box = box
                continue

            gap = box.x1 - last_box.x2
            if (
                gap <= in_line_gap
                and y_overlap_ratio(last_box, box) >= same_line_overlap
            ):
                cur_boxes.append(box)
                cur_texts.append(text)
                last_box = box
            else:
                merged.append(MergedRegion(bbox=merge_bbox(
                    cur_boxes), text="".join(cur_texts)))
                cur_boxes = [box]
                cur_texts = [text]
                last_box = box

        if cur_boxes:
            merged.append(MergedRegion(bbox=merge_bbox(
                cur_boxes), text="".join(cur_texts)))

    # 3) 行间合并：把多行对话框合并成一段（纵向距离小且 x 方向重叠足够）
    merged.sort(key=lambda r: (r.bbox.y1, r.bbox.x1))

    merged = _merge_between_lines(
        merged,
        merge_gap_between_lines_px=between_line_gap,
        x_overlap_threshold=0.5,
    )

    # 4) 最终顺序：从上到下、从左到右
    merged.sort(key=lambda r: (r.bbox.y1, r.bbox.x1))
    return merged


def _x_overlap_ratio(a: BBox, b: BBox) -> float:
    left = max(a.x1, b.x1)
    right = min(a.x2, b.x2)
    overlap = max(0, right - left)
    denom = min(a.w, b.w)
    return overlap / denom if denom > 0 else 0.0


def _merge_between_lines(
    items: list[MergedRegion],
    *,
    merge_gap_between_lines_px: int,
    x_overlap_threshold: float,
) -> list[MergedRegion]:
    if not items:
        return []

    out: list[MergedRegion] = []
    cur = items[0]
    for nxt in items[1:]:
        v_gap = max(0, int(nxt.bbox.y1) - int(cur.bbox.y2))
        if (
            v_gap <= int(merge_gap_between_lines_px)
            and _x_overlap_ratio(cur.bbox, nxt.bbox)
            >= float(x_overlap_threshold)
        ):
            cur = MergedRegion(
                bbox=merge_bbox([cur.bbox, nxt.bbox]),
                text=f"{cur.text}{nxt.text}",
            )
        else:
            out.append(cur)
            cur = nxt
    out.append(cur)
    return out
