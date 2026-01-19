from __future__ import annotations

from dataclasses import dataclass

from manga_ocr import MangaOcr
from PIL import Image

from autotransgal.util.geometry import BBox


@dataclass(frozen=True)
class RecognizedRegion:
    bbox: BBox
    text: str


class TextRecognizer:
    def __init__(self, force_cpu: bool = False):
        # manga-ocr 初始化很重，必须复用
        self._ocr = MangaOcr(
            pretrained_model_name_or_path="kha-white/manga-ocr-base",
            force_cpu=force_cpu,
        )

    def recognize(self, image: Image.Image, bbox: BBox) -> str:
        crop = image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        text = self._ocr(crop)
        return (text or "").strip()
