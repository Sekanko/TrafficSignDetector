"""Unauthenticated HTML admin panel for managing mobile model bundles."""

from __future__ import annotations

from html import escape
from pathlib import Path
from string import Template

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from src.api.models import (
    DEFAULT_CLASSIFIER_INPUT_SIZE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_INPUT_SIZE,
    MobileModel,
    create_pipeline_model,
    delete_mobile_model,
    get_mobile_model,
    list_mobile_models,
    update_pipeline_model,
)

router = APIRouter()
API_DIR = Path(__file__).parent
TEMPLATES_DIR = API_DIR / "templates"
STATIC_DIR = API_DIR / "static"


@router.get("/admin", response_class=HTMLResponse)
def admin_panel(created: str | None = None, updated: str | None = None, deleted: str | None = None) -> str:
    models = list_mobile_models()
    return _render_template(
        "admin.html",
        title="TrafficSignDetector Admin",
        messages=_messages(created=created, updated=updated, deleted=deleted),
        model_rows="\n".join(_model_row(model) for model in models)
        or '<tr><td colspan="4">Brak modeli.</td></tr>',
        default_confidence=DEFAULT_CONFIDENCE_THRESHOLD,
        default_detector_input=DEFAULT_INPUT_SIZE,
        default_classifier_input=DEFAULT_CLASSIFIER_INPUT_SIZE,
    )


@router.get("/admin/models/{model_id}/edit", response_class=HTMLResponse)
def admin_edit_panel(model_id: str) -> str:
    model = get_mobile_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    detector_file = model.file_metadata.get("detector", {})
    classifier_file = model.file_metadata.get("classifier", {})
    return _render_template(
        "edit_model.html",
        title=f"Edytuj {model.name}",
        model_id=escape(model.id),
        model_name=escape(model.name),
        model_version=escape(model.version),
        confidence=model.confidence_threshold,
        detector_input=detector_file.get("inputSize", model.input_size),
        classifier_input=classifier_file.get(
            "inputSize",
            DEFAULT_CLASSIFIER_INPUT_SIZE,
        ),
    )


@router.post("/admin/models")
def create_model_from_upload(
    name: str = Form(...),
    version: str = Form(...),
    confidence_threshold: float = Form(DEFAULT_CONFIDENCE_THRESHOLD),
    detector_input_size: int = Form(DEFAULT_INPUT_SIZE),
    classifier_input_size: int = Form(DEFAULT_CLASSIFIER_INPUT_SIZE),
    detector_file: UploadFile = File(...),
    classifier_file: UploadFile = File(...),
) -> RedirectResponse:
    model_name, model_version = _validate_name_version(name, version)
    _validate_tflite_upload(detector_file, "detector")
    _validate_tflite_upload(classifier_file, "classifier")

    try:
        model = create_pipeline_model(
            name=model_name,
            version=model_version,
            detector_file=detector_file.file,
            classifier_file=classifier_file.file,
            detector_input_size=detector_input_size,
            classifier_input_size=classifier_input_size,
            confidence_threshold=confidence_threshold,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return RedirectResponse(url=f"/admin?created={model.id}", status_code=303)


@router.post("/admin/models/{model_id}/edit")
def edit_model_from_admin(
    model_id: str,
    name: str = Form(...),
    version: str = Form(...),
    confidence_threshold: float = Form(DEFAULT_CONFIDENCE_THRESHOLD),
    detector_input_size: int = Form(DEFAULT_INPUT_SIZE),
    classifier_input_size: int = Form(DEFAULT_CLASSIFIER_INPUT_SIZE),
    detector_file: UploadFile | None = File(None),
    classifier_file: UploadFile | None = File(None),
) -> RedirectResponse:
    model_name, model_version = _validate_name_version(name, version)
    detector_stream = _optional_tflite_stream(detector_file, "detector")
    classifier_stream = _optional_tflite_stream(classifier_file, "classifier")

    try:
        model = update_pipeline_model(
            model_id=model_id,
            name=model_name,
            version=model_version,
            detector_file=detector_stream,
            classifier_file=classifier_stream,
            detector_input_size=detector_input_size,
            classifier_input_size=classifier_input_size,
            confidence_threshold=confidence_threshold,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return RedirectResponse(url=f"/admin?updated={model.id}", status_code=303)


@router.post("/admin/models/{model_id}/delete")
def delete_model_from_admin(model_id: str) -> RedirectResponse:
    try:
        delete_mobile_model(model_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return RedirectResponse(url=f"/admin?deleted={model_id}", status_code=303)


def _render_template(template_name: str, **values: object) -> str:
    template = Template((TEMPLATES_DIR / template_name).read_text(encoding="utf-8"))
    return template.safe_substitute({key: str(value) for key, value in values.items()})


def _messages(
    *,
    created: str | None,
    updated: str | None,
    deleted: str | None,
) -> str:
    messages = []
    if created:
        messages.append(f'Utworzono model: <code>{escape(created)}</code>')
    if updated:
        messages.append(f'Zaktualizowano model: <code>{escape(updated)}</code>')
    if deleted:
        messages.append(f'Usunięto model: <code>{escape(deleted)}</code>')

    return "\n".join(f'<p class="success">{message}</p>' for message in messages)


def _model_row(model: MobileModel) -> str:
    files = "\n".join(
        f'<div class="file-chip" title="{escape(path.name)}">'
        f'<span>{escape(role)}</span><code>{escape(path.name)}</code></div>'
        for role, path in model.file_paths.items()
    )
    model_id = escape(model.id)
    return f"""<tr>
  <td>{escape(model.name)}</td>
  <td>{escape(model.version)}</td>
  <td>{files}</td>
  <td class="actions">
    <a class="button-link" href="/admin/models/{model_id}/edit">Edytuj</a>
    <form action="/admin/models/{model_id}/delete" method="post" onsubmit="return confirm('Usunąć ten model?');">
      <button class="danger" type="submit">Usuń</button>
    </form>
  </td>
</tr>"""


def _validate_name_version(name: str, version: str) -> tuple[str, str]:
    model_name = name.strip()
    model_version = version.strip()
    if not model_name or not model_version:
        raise HTTPException(status_code=400, detail="Model name and version are required")
    return model_name, model_version


def _optional_tflite_stream(upload: UploadFile | None, role: str):
    if upload is None or not upload.filename:
        return None

    _validate_tflite_upload(upload, role)
    return upload.file


def _validate_tflite_upload(upload: UploadFile, role: str) -> None:
    filename = upload.filename or ""
    if not filename.endswith(".tflite"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded {role} file must be a .tflite file",
        )
