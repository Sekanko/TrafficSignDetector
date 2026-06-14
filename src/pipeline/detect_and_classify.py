"""Combine the YOLO sign detector with a trained classifier and visualize results.

YOLO proposes candidate boxes on full street-scene images. By default this
uses the COCO-pretrained YOLO11n class-agnostically: any box above --det-conf
is treated as "something that might be a sign" (see src/detection/evaluate.py
for the same reasoning). Pass --detector-weights pointing at a fine-tuned
checkpoint from src/detection/train.py (under models/detection/) to use the
single-class traffic_sign detector instead. Each box is cropped, expanded by
a small margin, and classified with one of the trained models from
src/classification/models/ to predict the concrete sign type.

For each sampled image, saves a side-by-side comparison (ground truth boxes
vs. detector+classifier boxes) to outputs/pipeline_demo/.

Usage:
    python -m src.pipeline.detect_and_classify
    python -m src.pipeline.detect_and_classify --classifier custom_cnn --sample-count 4
    python -m src.pipeline.detect_and_classify --detector-weights models/detection/<run>/weights/best.pt
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ultralytics import YOLO

from src.classification.models import MODEL_NAMES, build_model
from src.classification.train import IMAGENET_MEAN, IMAGENET_STD
from src.data_prep.paths import PROJECT_ROOT
from src.data_prep.taxonomy import Taxonomy, load_taxonomy
from src.detection.constants import IMAGE_EXTENSIONS
from src.detection.dataset import load_ground_truth, split_image_paths
from src.detection.predict import predict_boxes
from src.detection.types import Box

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_demo"


@dataclass(frozen=True)
class SignPrediction:
    box: Box
    detection_confidence: float
    class_name: str
    class_name_pl: str
    class_confidence: float


def find_latest_checkpoint(model_name: str) -> Path:
    candidates = sorted((PROJECT_ROOT / "models").glob(f"{model_name}_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for model '{model_name}' in models/. "
            f"Train one first with `python -m src.classification.train --model {model_name} ...`."
        )
    return candidates[-1]


def load_classifier(model_name: str, checkpoint: Path, num_classes: int, device: str) -> torch.nn.Module:
    model = build_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model


def classification_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def expand_box(box: Box, margin_ratio: float, img_width: int, img_height: int) -> Box:
    dx = (box.x2 - box.x1) * margin_ratio
    dy = (box.y2 - box.y1) * margin_ratio
    return Box(
        x1=max(0.0, box.x1 - dx),
        y1=max(0.0, box.y1 - dy),
        x2=min(float(img_width), box.x2 + dx),
        y2=min(float(img_height), box.y2 + dy),
    )


@torch.no_grad()
def classify_crop(model: torch.nn.Module, crop: Image.Image, transform: transforms.Compose, device: str) -> tuple[int, float]:
    tensor = transform(crop).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1)
    confidence, class_id = probs.max(dim=1)
    return int(class_id.item()), float(confidence.item())


def run_pipeline(
    image_paths: list[Path],
    *,
    detector: YOLO,
    classifier: torch.nn.Module,
    taxonomy: Taxonomy,
    transform: transforms.Compose,
    device: str,
    img_size: int,
    det_conf: float,
    crop_margin: float,
) -> dict[Path, list[SignPrediction]]:
    raw_predictions = predict_boxes(
        detector,
        image_paths,
        img_size=img_size,
        conf=det_conf,
        max_det=50,
        device=device,
        batch_size=len(image_paths),
    )

    results: dict[Path, list[SignPrediction]] = {}
    for image_path, predictions in raw_predictions.items():
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")

        signs = []
        for prediction in predictions:
            crop_box = expand_box(prediction.box, crop_margin, image.width, image.height)
            crop = image.crop((crop_box.x1, crop_box.y1, crop_box.x2, crop_box.y2))
            if crop.width == 0 or crop.height == 0:
                continue
            class_id, class_confidence = classify_crop(classifier, crop, transform, device)
            taxonomy_class = taxonomy.by_id(class_id)
            signs.append(
                SignPrediction(
                    box=prediction.box,
                    detection_confidence=prediction.confidence,
                    class_name=taxonomy_class.name,
                    class_name_pl=taxonomy_class.pl,
                    class_confidence=class_confidence,
                )
            )
        results[image_path] = signs
    return results


def _annotated_panel(
    image: Image.Image,
    boxes: list[Box],
    labels: list[str] | None,
    color: str,
    title: str,
    panel_width: int,
) -> Image.Image:
    """Resize `image` to `panel_width` first, then draw boxes/labels on the
    resized copy with a font sized for that resolution - drawing at full
    resolution and shrinking afterwards makes text unreadably small."""
    ratio = panel_width / image.width
    resized = image.resize((panel_width, max(1, round(image.height * ratio))))

    font_size = max(14, panel_width // 32)
    font = ImageFont.load_default(size=font_size)
    title_height = font_size + 12
    line_width = max(2, panel_width // 250)

    panel = Image.new("RGB", (panel_width, resized.height + title_height), "white")
    panel.paste(resized, (0, title_height))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 6), title, fill="black", font=font)

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box.x1 * ratio, box.y1 * ratio + title_height, box.x2 * ratio, box.y2 * ratio + title_height
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        if labels is not None and index < len(labels):
            text = labels[index]
            label_y = max(title_height, y1 - font_size - 4)
            left, top, right, bottom = draw.textbbox((x1, label_y), text, font=font)
            draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill=color)
            draw.text((x1, label_y), text, fill="white", font=font)

    return panel


def save_comparison(
    image_path: Path,
    *,
    ground_truth: list[Box],
    predictions: list[SignPrediction],
    output_path: Path,
    panel_width: int = 640,
) -> None:
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")

    gt_panel = _annotated_panel(image, ground_truth, None, "green", "Ground truth", panel_width)

    labels = [f"{p.class_name} {p.class_confidence:.2f} (det {p.detection_confidence:.2f})" for p in predictions]
    pred_panel = _annotated_panel(image, [p.box for p in predictions], labels, "red", "YOLO + classifier", panel_width)

    height = max(gt_panel.height, pred_panel.height)
    combined = Image.new("RGB", (panel_width * 2, height), "white")
    combined.paste(gt_panel, (0, 0))
    combined.paste(pred_panel, (panel_width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path, quality=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detector + classifier on sample images")
    parser.add_argument("--detector-weights", default="yolo11n.pt")
    parser.add_argument("--classifier", default="mobilenet_v2", choices=MODEL_NAMES)
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None, help="Directory of images (default: data/processed/detection/images/<split>)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--img-size", type=int, default=640, help="YOLO inference resolution")
    parser.add_argument("--det-conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--crop-margin", type=float, default=0.1, help="Crop margin around each detection, as a fraction of box size")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = load_taxonomy()

    checkpoint = args.classifier_checkpoint or find_latest_checkpoint(args.classifier)
    classifier = load_classifier(args.classifier, checkpoint, len(taxonomy), args.device)
    transform = classification_transform()

    detector = YOLO(args.detector_weights)

    if args.source is None:
        image_paths = split_image_paths(args.split)
    else:
        image_paths = sorted(path for path in args.source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

    sample = random.Random(args.seed).sample(image_paths, min(args.sample_count, len(image_paths)))

    results = run_pipeline(
        sample,
        detector=detector,
        classifier=classifier,
        taxonomy=taxonomy,
        transform=transform,
        device=args.device,
        img_size=args.img_size,
        det_conf=args.det_conf,
        crop_margin=args.crop_margin,
    )

    detector_path = Path(args.detector_weights)
    if "detection" in detector_path.parts:
        detector_label = "fine-tuned single-class traffic_sign detector"
    else:
        detector_label = "COCO-pretrained, class-agnostic boxes"

    print("=" * 72)
    print(f"Detector:   {args.detector_weights} ({detector_label})")
    print(f"Classifier: {args.classifier} ({checkpoint.name})")
    print(f"Images:     {len(sample)} from '{args.split}' split (seed={args.seed})")
    print("=" * 72)

    for image_path in sample:
        predictions = results[image_path]
        ground_truth = load_ground_truth(image_path, args.split)
        output_path = args.output_dir / f"{image_path.stem}_comparison.jpg"
        save_comparison(image_path, ground_truth=ground_truth, predictions=predictions, output_path=output_path)

        print(f"\n{image_path.name}  (ground truth boxes: {len(ground_truth)})")
        if not predictions:
            print("  no detections above --det-conf")
        for prediction in predictions:
            print(
                f"  -> {prediction.class_name_pl} ({prediction.class_name}) "
                f"cls={prediction.class_confidence:.2f} det={prediction.detection_confidence:.2f}"
            )
        print(f"  saved: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
