"""Fine-tune YOLO11n as a one-class traffic sign detector.

Usage:
    python -m src.detection.train --epochs 50 --img-size 320 --batch-size 16
    python -m src.detection.train --epochs 50 --export-tflite --tflite-int8
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from src.data_prep.paths import PROJECT_ROOT
from src.detection.cli import parse_args
from src.detection.constants import DETECTION_MODELS_DIR
from src.detection.dataset import ensure_dataset_exists
from src.detection.evaluate import evaluate_model, metric_delta
from src.detection.export import export_tflite
from src.detection.utils import project_relative
from src.detection.visualize import save_visual_comparison


def train_yolo11n(args, dataset_yaml: Path, run_name: str):
    train_model = YOLO(args.base_model)
    train_kwargs = {
        "data": str(dataset_yaml),
        "epochs": args.epochs,
        "imgsz": args.img_size,
        "batch": args.batch_size,
        "lr0": args.lr,
        "patience": args.patience,
        "workers": args.workers,
        "seed": args.seed,
        "single_cls": True,
        "project": str(DETECTION_MODELS_DIR),
        "name": run_name,
        "exist_ok": False,
        "plots": True,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    return train_model.train(**train_kwargs)


def build_summary(
    *,
    args,
    dataset_yaml: Path,
    run_name: str,
    save_dir: Path,
    best_weights: Path,
    last_weights: Path,
    baseline_metrics,
    fine_tuned_metrics,
    exported_tflite: str | None,
    visualization_files: list[str],
) -> dict:
    return {
        "run_name": run_name,
        "model": "YOLO11n",
        "base_model": args.base_model,
        "task": "one-class traffic sign detection",
        "dataset": project_relative(dataset_yaml),
        "compare_split": args.compare_split,
        "epochs": args.epochs,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "patience": args.patience,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "confidence_threshold_for_visualizations": args.conf,
        "eval_confidence_threshold": args.eval_conf,
        "baseline_pretrained": asdict(baseline_metrics),
        "fine_tuned": asdict(fine_tuned_metrics),
        "delta_fine_tuned_minus_baseline": metric_delta(fine_tuned_metrics, baseline_metrics),
        "best_weights": project_relative(best_weights),
        "last_weights": project_relative(last_weights) if last_weights.exists() else None,
        "exported_tflite": exported_tflite,
        "ultralytics_run_dir": project_relative(save_dir),
        "visualizations": visualization_files,
    }


def print_run_summary(args, best_weights: Path, summary_path: Path, exported_tflite: str | None, baseline_metrics, fine_tuned_metrics, visualizations_dir: Path) -> None:
    print()
    print("=" * 72)
    print("YOLO11n traffic sign detector")
    print(f"Best weights:    {best_weights}")
    print(f"Summary:         {summary_path}")
    if exported_tflite is not None:
        print(f"TFLite export:   {PROJECT_ROOT / exported_tflite}")
    print()
    print("Class-agnostic localization metrics on split:", args.compare_split)
    print(f"Baseline mAP50:  {baseline_metrics.map50:.4f}")
    print(f"Fine-tuned mAP50:{fine_tuned_metrics.map50:.4f}")
    print(f"Delta mAP50:     {fine_tuned_metrics.map50 - baseline_metrics.map50:+.4f}")
    print(f"Baseline R@50:   {baseline_metrics.recall_at_50:.4f}")
    print(f"Fine-tuned R@50: {fine_tuned_metrics.recall_at_50:.4f}")
    print(f"Visualizations:  {visualizations_dir}")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    dataset_yaml = ensure_dataset_exists()

    run_name = f"yolo11n_traffic_sign_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = DETECTION_MODELS_DIR / run_name
    visualizations_dir = run_dir / "visualizations"
    DETECTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Evaluating YOLO11n before fine-tuning...")
    baseline_model = YOLO(args.base_model)
    baseline_metrics, baseline_predictions, ground_truths = evaluate_model(
        baseline_model,
        split=args.compare_split,
        img_size=args.img_size,
        conf=args.eval_conf,
        max_det=args.max_det,
        device=args.device,
        batch_size=args.batch_size,
    )

    print("Fine-tuning YOLO11n...")
    train_results = train_yolo11n(args, dataset_yaml, run_name)
    save_dir = Path(train_results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Expected best weights were not created: {best_weights}")

    print("Evaluating fine-tuned YOLO11n...")
    fine_tuned_model = YOLO(best_weights)
    fine_tuned_metrics, fine_tuned_predictions, _ = evaluate_model(
        fine_tuned_model,
        split=args.compare_split,
        img_size=args.img_size,
        conf=args.eval_conf,
        max_det=args.max_det,
        device=args.device,
        batch_size=args.batch_size,
    )

    print("Saving qualitative comparison visualizations...")
    visualization_files = save_visual_comparison(
        split=args.compare_split,
        baseline_predictions=baseline_predictions,
        fine_tuned_predictions=fine_tuned_predictions,
        ground_truths=ground_truths,
        output_dir=visualizations_dir,
        sample_count=args.sample_count,
        confidence_threshold=args.conf,
        seed=args.seed,
    )

    exported_tflite = export_tflite(fine_tuned_model, args, dataset_yaml)
    summary = build_summary(
        args=args,
        dataset_yaml=dataset_yaml,
        run_name=run_name,
        save_dir=save_dir,
        best_weights=best_weights,
        last_weights=last_weights,
        baseline_metrics=baseline_metrics,
        fine_tuned_metrics=fine_tuned_metrics,
        exported_tflite=exported_tflite,
        visualization_files=visualization_files,
    )
    summary_path = save_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_run_summary(
        args,
        best_weights,
        summary_path,
        exported_tflite,
        baseline_metrics,
        fine_tuned_metrics,
        visualizations_dir,
    )


if __name__ == "__main__":
    main()
