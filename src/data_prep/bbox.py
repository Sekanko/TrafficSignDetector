"""Bounding box helpers shared by all dataset adapters and build scripts."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class BBox:
    """Pixel-space bounding box (x1, y1) inclusive, (x2, y2) exclusive."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def clip(self, img_width: int, img_height: int) -> BBox:
        return BBox(
            x1=max(0, min(self.x1, img_width)),
            y1=max(0, min(self.y1, img_height)),
            x2=max(0, min(self.x2, img_width)),
            y2=max(0, min(self.y2, img_height)),
        )

    def expand(self, margin_ratio: float, img_width: int, img_height: int) -> BBox:
        """Grow the box by `margin_ratio` of its size on each side, then clip
        to the image bounds."""
        dx = round(self.width * margin_ratio)
        dy = round(self.height * margin_ratio)
        return BBox(
            x1=self.x1 - dx,
            y1=self.y1 - dy,
            x2=self.x2 + dx,
            y2=self.y2 + dy,
        ).clip(img_width, img_height)

    def to_yolo(self, img_width: int, img_height: int) -> tuple[float, float, float, float]:
        """Convert to normalized (center_x, center_y, width, height)."""
        cx = (self.x1 + self.x2) / 2 / img_width
        cy = (self.y1 + self.y2) / 2 / img_height
        w = self.width / img_width
        h = self.height / img_height
        return cx, cy, w, h

    @classmethod
    def from_yolo(
        cls, cx: float, cy: float, w: float, h: float, img_width: int, img_height: int
    ) -> BBox:
        """Inverse of `to_yolo`: normalized (center_x, center_y, width, height)
        -> pixel-space box."""
        return cls(
            x1=round((cx - w / 2) * img_width),
            y1=round((cy - h / 2) * img_height),
            x2=round((cx + w / 2) * img_width),
            y2=round((cy + h / 2) * img_height),
        )


def crop_image(image: Image.Image, bbox: BBox) -> Image.Image:
    return image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
