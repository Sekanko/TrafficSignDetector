"""Qualitative visual comparisons for detector runs."""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

from src.detection.types import Box, Prediction
from src.detection.utils import project_relative


def draw_boxes(
    image: Image.Image,
    boxes: list[Box],
    *,
    color: str,
    labels: list[str] | None = None,
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for index, box in enumerate(boxes):
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=3)
        if labels is not None and index < len(labels):
            text = labels[index]
            left, top, right, bottom = draw.textbbox((box.x1, box.y1), text)
            draw.rectangle((left, max(0, top - 2), right + 4, bottom + 2), fill=color)
            draw.text((box.x1 + 2, max(0, box.y1 - 1)), text, fill="white")
    return annotated


def titled_panel(image: Image.Image, title: str) -> Image.Image:
    title_height = 28
    panel = Image.new("RGB", (image.width, image.height + title_height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), title, fill="black")
    panel.paste(image, (0, title_height))
    return panel


def resize_for_grid(image: Image.Image, width: int) -> Image.Image:
    ratio = width / image.width
    return image.resize((width, max(1, math.floor(image.height * ratio))))


def build_comparison_panels(
    image_path: Path,
    *,
    baseline_predictions: dict[Path, list[Prediction]],
    fine_tuned_predictions: dict[Path, list[Prediction]],
    ground_truths: dict[Path, list[Box]],
    confidence_threshold: float,
    panel_width: int,
    detector_name: str,
) -> list[Image.Image]:
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")

    baseline = [
        prediction
        for prediction in baseline_predictions.get(image_path, [])
        if prediction.confidence >= confidence_threshold
    ]
    fine_tuned = [
        prediction
        for prediction in fine_tuned_predictions.get(image_path, [])
        if prediction.confidence >= confidence_threshold
    ]

    return [
        titled_panel(
            resize_for_grid(draw_boxes(image, ground_truths[image_path], color="green"), panel_width),
            "Ground truth",
        ),
        titled_panel(
            resize_for_grid(
                draw_boxes(
                    image,
                    [prediction.box for prediction in baseline],
                    color="orange",
                    labels=[f"{prediction.confidence:.2f}" for prediction in baseline],
                ),
                panel_width,
            ),
            f"{detector_name} pretrained",
        ),
        titled_panel(
            resize_for_grid(
                draw_boxes(
                    image,
                    [prediction.box for prediction in fine_tuned],
                    color="blue",
                    labels=[f"{prediction.confidence:.2f}" for prediction in fine_tuned],
                ),
                panel_width,
            ),
            f"{detector_name} fine-tuned",
        ),
    ]


def save_overview(
    *,
    split: str,
    selected: list[Path],
    baseline_predictions: dict[Path, list[Prediction]],
    fine_tuned_predictions: dict[Path, list[Prediction]],
    ground_truths: dict[Path, list[Box]],
    output_dir: Path,
    confidence_threshold: float,
    panel_width: int,
    detector_name: str,
) -> str | None:
    if not selected:
        return None

    rows = [
        build_comparison_panels(
            image_path,
            baseline_predictions=baseline_predictions,
            fine_tuned_predictions=fine_tuned_predictions,
            ground_truths=ground_truths,
            confidence_threshold=confidence_threshold,
            panel_width=panel_width,
            detector_name=detector_name,
        )
        for image_path in selected
    ]
    row_heights = [max(panel.height for panel in row) for row in rows]
    overview = Image.new(
        "RGB",
        (panel_width * 3, sum(row_heights)),
        "white",
    )

    y_offset = 0
    for row, row_height in zip(rows, row_heights):
        for panel_index, panel in enumerate(row):
            overview.paste(panel, (panel_width * panel_index, y_offset))
        y_offset += row_height

    output_path = output_dir / f"{split}_comparison_overview_{len(selected)}samples.jpg"
    overview.save(output_path, quality=92)
    return project_relative(output_path)


def save_visual_comparison(
    *,
    split: str,
    baseline_predictions: dict[Path, list[Prediction]],
    fine_tuned_predictions: dict[Path, list[Prediction]],
    ground_truths: dict[Path, list[Box]],
    output_dir: Path,
    sample_count: int,
    confidence_threshold: float,
    seed: int,
    detector_name: str,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [path for path, boxes in ground_truths.items() if boxes]
    random.Random(seed).shuffle(candidates)
    selected = candidates[:sample_count]
    saved_files = []

    for index, image_path in enumerate(selected, start=1):
        panel_width = 360
        panels = build_comparison_panels(
            image_path,
            baseline_predictions=baseline_predictions,
            fine_tuned_predictions=fine_tuned_predictions,
            ground_truths=ground_truths,
            confidence_threshold=confidence_threshold,
            panel_width=panel_width,
            detector_name=detector_name,
        )
        height = max(panel.height for panel in panels)
        comparison = Image.new("RGB", (panel_width * len(panels), height), "white")
        for panel_index, panel in enumerate(panels):
            comparison.paste(panel, (panel_width * panel_index, 0))

        output_path = output_dir / f"{split}_comparison_{index:02d}_{image_path.stem}.jpg"
        comparison.save(output_path, quality=92)
        saved_files.append(project_relative(output_path))

    overview = save_overview(
        split=split,
        selected=selected[:5],
        baseline_predictions=baseline_predictions,
        fine_tuned_predictions=fine_tuned_predictions,
        ground_truths=ground_truths,
        output_dir=output_dir,
        confidence_threshold=confidence_threshold,
        panel_width=360,
        detector_name=detector_name,
    )
    if overview is not None:
        saved_files.insert(0, overview)

    return saved_files
