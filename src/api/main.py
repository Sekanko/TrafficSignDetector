"""FastAPI app exposing TFLite models to the Flutter client.

Run locally:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.admin import STATIC_DIR, router as admin_router
from src.api.models import (
    API_MODELS_DIR,
    MobileModel,
    get_mobile_model,
    list_mobile_models,
)

app = FastAPI(title="TrafficSignDetector Models API")
ANDROID_EMULATOR_HOST = "10.0.2.2"
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def models(request: Request) -> dict[str, list[dict[str, object]]]:
    return {
        "models": [_model_response(model, request) for model in list_mobile_models()]
    }


@app.get("/models/{model_id}/download")
def download_model(model_id: str) -> FileResponse:
    model = get_mobile_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return FileResponse(
        model.file_path,
        media_type="application/octet-stream",
        filename=model.file_name,
    )


@app.get("/models/{model_id}/files/{file_role}/download")
def download_model_file(model_id: str, file_role: str) -> FileResponse:
    model = get_mobile_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    file_path = model.file_paths.get(file_role)
    if file_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"File role '{file_role}' not found for model '{model_id}'",
        )

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=file_path.name,
    )


def _model_response(model: MobileModel, request: Request) -> dict[str, object]:
    base_url = _client_base_url(request)
    model_files = {
        role: {
            **model.file_metadata.get(role, {}),
            "path": file_path.name,
            "downloadUrl": f"{base_url}/models/{model.id}/files/{role}/download",
        }
        for role, file_path in model.file_paths.items()
    }

    return {
        "id": model.id,
        "name": model.name,
        "version": model.version,
        "modelType": model.model_type,
        "path": model.file_name,
        "downloadUrl": f"{base_url}/models/{model.id}/download",
        "modelFiles": model_files,
        "labels": model.labels,
        "inputSize": model.input_size,
        "confidenceThreshold": model.confidence_threshold,
        "isDownloaded": False,
        "isSelected": False,
        "isAvailableInApi": True,
    }


def _client_base_url(request: Request) -> str:
    base_url = str(request.base_url).rstrip("/")
    if request.url.hostname == ANDROID_EMULATOR_HOST:
        return base_url.replace("127.0.0.1", ANDROID_EMULATOR_HOST).replace(
            "localhost",
            ANDROID_EMULATOR_HOST,
        )
    return base_url


if __name__ == "__main__":
    import uvicorn

    API_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
