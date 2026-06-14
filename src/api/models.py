"""Model registry helpers used by the mobile models API."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from src.data_prep.paths import PROJECT_ROOT

API_MODELS_DIR = PROJECT_ROOT / "models" / "api"
DEFAULT_INPUT_SIZE = 320
DEFAULT_CONFIDENCE_THRESHOLD = 0.45
DEFAULT_CLASSIFIER_INPUT_SIZE = 224


@dataclass(frozen=True)
class MobileModel:
    id: str
    name: str
    version: str
    file_paths: dict[str, Path]
    labels: list[str]
    file_metadata: dict[str, dict[str, object]]
    model_type: str = "single_tflite"
    input_size: int = DEFAULT_INPUT_SIZE
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD

    @property
    def primary_role(self) -> str:
        return next(iter(self.file_paths))

    @property
    def file_path(self) -> Path:
        return self.file_paths[self.primary_role]

    @property
    def file_name(self) -> str:
        return self.file_path.name


def _taxonomy_labels() -> list[str]:
    try:
        from src.data_prep.taxonomy import load_taxonomy
    except ModuleNotFoundError:
        return _taxonomy_labels_from_yaml_text()

    return [taxonomy_class.pl for taxonomy_class in load_taxonomy().classes]


def _taxonomy_labels_from_yaml_text() -> list[str]:
    taxonomy_path = PROJECT_ROOT / "configs" / "taxonomy.yaml"
    if not taxonomy_path.exists():
        return []

    labels: list[str] = []
    for line in taxonomy_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("pl:"):
            continue

        label = stripped.split(":", 1)[1].strip()
        if len(label) >= 2 and label[0] == label[-1] and label[0] in {"'", '"'}:
            label = label[1:-1]
        labels.append(label)

    return labels


def _read_sidecar_metadata(model_path: Path) -> dict[str, Any]:
    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        decoded = json.load(metadata_file)

    if not isinstance(decoded, dict):
        raise ValueError(f"Metadata file must contain a JSON object: {metadata_path}")

    return decoded


def _metadata_value(
    metadata: dict[str, Any],
    camel_case: str,
    snake_case: str,
    default: Any,
) -> Any:
    return metadata.get(camel_case, metadata.get(snake_case, default))


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "traffic_sign_pipeline"


def _copy_tflite(upload: BinaryIO, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as output_file:
        shutil.copyfileobj(upload, output_file)


def create_pipeline_model(
    *,
    name: str,
    version: str,
    detector_file: BinaryIO,
    classifier_file: BinaryIO,
    detector_input_size: int = DEFAULT_INPUT_SIZE,
    classifier_input_size: int = DEFAULT_CLASSIFIER_INPUT_SIZE,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    models_dir: Path = API_MODELS_DIR,
) -> MobileModel:
    """Create or replace a detector+classifier bundle exposed by the mobile API."""
    model_id = _slug(f"{name}_{version}")
    bundle_dir = models_dir / model_id
    detector_path = bundle_dir / "detector.tflite"
    classifier_path = bundle_dir / "classifier.tflite"

    _copy_tflite(detector_file, detector_path)
    _copy_tflite(classifier_file, classifier_path)

    metadata = {
        "id": model_id,
        "name": name,
        "version": version,
        "modelType": "detector_classifier_pipeline",
        "inputSize": detector_input_size,
        "confidenceThreshold": confidence_threshold,
        "files": {
            "detector": {
                "path": detector_path.name,
                "inputSize": detector_input_size,
                "outputFormat": "yolo",
            },
            "classifier": {
                "path": classifier_path.name,
                "inputSize": classifier_input_size,
                "outputFormat": "classification_logits",
            },
        },
    }
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with open(bundle_dir / "model.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        metadata_file.write("\n")

    model = _build_bundle_model(bundle_dir)
    if model is None:
        raise ValueError(f"Could not create model bundle: {bundle_dir}")
    return model


def update_pipeline_model(
    *,
    model_id: str,
    name: str,
    version: str,
    detector_file: BinaryIO | None = None,
    classifier_file: BinaryIO | None = None,
    detector_input_size: int = DEFAULT_INPUT_SIZE,
    classifier_input_size: int = DEFAULT_CLASSIFIER_INPUT_SIZE,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    models_dir: Path = API_MODELS_DIR,
) -> MobileModel:
    """Update an existing detector+classifier bundle and optionally replace files."""
    bundle_dir = models_dir / model_id
    if not bundle_dir.exists():
        raise ValueError(f"Model bundle does not exist: {model_id}")

    metadata = _read_bundle_metadata(bundle_dir)
    raw_files = metadata.get("files")
    if not isinstance(raw_files, dict):
        raise ValueError(f"Bundle metadata must contain a 'files' object: {bundle_dir}")

    detector_path = bundle_dir / _file_name_for_role(raw_files, "detector", "detector.tflite")
    classifier_path = bundle_dir / _file_name_for_role(
        raw_files,
        "classifier",
        "classifier.tflite",
    )

    if detector_file is not None:
        _copy_tflite(detector_file, detector_path)
    if classifier_file is not None:
        _copy_tflite(classifier_file, classifier_path)

    metadata.update(
        {
            "id": model_id,
            "name": name,
            "version": version,
            "modelType": "detector_classifier_pipeline",
            "inputSize": detector_input_size,
            "confidenceThreshold": confidence_threshold,
            "files": {
                "detector": {
                    "path": detector_path.name,
                    "inputSize": detector_input_size,
                    "outputFormat": "yolo",
                },
                "classifier": {
                    "path": classifier_path.name,
                    "inputSize": classifier_input_size,
                    "outputFormat": "classification_logits",
                },
            },
        }
    )

    with open(bundle_dir / "model.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        metadata_file.write("\n")

    model = _build_bundle_model(bundle_dir)
    if model is None:
        raise ValueError(f"Could not update model bundle: {bundle_dir}")
    return model


def delete_mobile_model(model_id: str, models_dir: Path = API_MODELS_DIR) -> None:
    """Delete a model exposed by the mobile API."""
    model = get_mobile_model(model_id, models_dir)
    if model is None:
        raise ValueError(f"Model does not exist: {model_id}")

    model_root = models_dir.resolve()
    file_path = model.file_path.resolve()

    if file_path.parent == model_root:
        file_path.unlink(missing_ok=True)
        file_path.with_suffix(".json").unlink(missing_ok=True)
        return

    bundle_dir = file_path.parent
    if model_root not in bundle_dir.parents and bundle_dir != model_root:
        raise ValueError(f"Refusing to delete model outside API directory: {model_id}")

    shutil.rmtree(bundle_dir)


def _file_name_for_role(
    raw_files: dict[str, Any],
    role: str,
    fallback: str,
) -> str:
    raw_file = raw_files.get(role)
    if isinstance(raw_file, dict):
        return Path(str(raw_file.get("path", fallback))).name
    if raw_file is not None:
        return Path(str(raw_file)).name
    return fallback


def _build_flat_model(model_path: Path) -> MobileModel:
    metadata = _read_sidecar_metadata(model_path)
    model_id = str(_metadata_value(metadata, "id", "id", model_path.stem))
    labels = metadata.get("labels")
    if labels is None:
        labels = _taxonomy_labels()

    return MobileModel(
        id=model_id,
        name=str(
            _metadata_value(
                metadata,
                "name",
                "name",
                model_path.stem.replace("_", " ").title(),
            )
        ),
        version=str(_metadata_value(metadata, "version", "version", "v1.0.0")),
        file_paths={"model": model_path},
        labels=[str(label) for label in labels],
        file_metadata={"model": {"path": model_path.name}},
        model_type=str(
            _metadata_value(metadata, "modelType", "model_type", "single_tflite")
        ),
        input_size=int(
            _metadata_value(metadata, "inputSize", "input_size", DEFAULT_INPUT_SIZE)
        ),
        confidence_threshold=float(
            _metadata_value(
                metadata,
                "confidenceThreshold",
                "confidence_threshold",
                DEFAULT_CONFIDENCE_THRESHOLD,
            )
        ),
    )


def _read_bundle_metadata(bundle_dir: Path) -> dict[str, Any]:
    metadata_path = bundle_dir / "model.json"
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        decoded = json.load(metadata_file)

    if not isinstance(decoded, dict):
        raise ValueError(f"Bundle metadata must contain a JSON object: {metadata_path}")

    return decoded


def _build_bundle_model(bundle_dir: Path) -> MobileModel | None:
    metadata = _read_bundle_metadata(bundle_dir)
    raw_files = metadata.get("files")
    if not isinstance(raw_files, dict):
        raise ValueError(f"Bundle metadata must contain a 'files' object: {bundle_dir}")

    file_paths: dict[str, Path] = {}
    file_metadata: dict[str, dict[str, object]] = {}
    for role, raw_file in raw_files.items():
        if isinstance(raw_file, dict):
            relative_path = raw_file.get("path")
            metadata_entry = dict(raw_file)
        else:
            relative_path = raw_file
            metadata_entry = {"path": str(raw_file)}

        if relative_path is None:
            raise ValueError(f"Missing path for file role '{role}' in {bundle_dir}")

        file_paths[str(role)] = bundle_dir / str(relative_path)
        file_metadata[str(role)] = metadata_entry

    if not file_paths or not all(path.exists() for path in file_paths.values()):
        return None

    labels = metadata.get("labels")
    if labels is None:
        labels = _taxonomy_labels()

    return MobileModel(
        id=str(metadata.get("id", bundle_dir.name)),
        name=str(metadata.get("name", bundle_dir.name.replace("_", " ").title())),
        version=str(metadata.get("version", "v1.0.0")),
        file_paths=file_paths,
        labels=[str(label) for label in labels],
        file_metadata=file_metadata,
        model_type=str(
            _metadata_value(
                metadata,
                "modelType",
                "model_type",
                "detector_classifier_pipeline",
            )
        ),
        input_size=int(
            _metadata_value(metadata, "inputSize", "input_size", DEFAULT_INPUT_SIZE)
        ),
        confidence_threshold=float(
            _metadata_value(
                metadata,
                "confidenceThreshold",
                "confidence_threshold",
                DEFAULT_CONFIDENCE_THRESHOLD,
            )
        ),
    )


def list_mobile_models(models_dir: Path = API_MODELS_DIR) -> list[MobileModel]:
    """Return every TFLite model available for download by the Flutter app."""
    if not models_dir.exists():
        return []

    flat_models = [
        _build_flat_model(model_path)
        for model_path in sorted(models_dir.glob("*.tflite"))
    ]
    bundle_models = [
        model
        for bundle_dir in sorted(path for path in models_dir.iterdir() if path.is_dir())
        if (model := _build_bundle_model(bundle_dir)) is not None
    ]
    return [*flat_models, *bundle_models]


def get_mobile_model(model_id: str, models_dir: Path = API_MODELS_DIR) -> MobileModel | None:
    for model in list_mobile_models(models_dir):
        if model.id == model_id:
            return model
    return None
