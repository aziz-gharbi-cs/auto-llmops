"""Model registry and lifecycle management endpoints (Section 22/26)."""

from fastapi import APIRouter, Depends, HTTPException, Request, status

from llmops.interfaces.api.schemas import ModelInfo, ModelsListResponse
from llmops.interfaces.api.security import Role, require_roles

router = APIRouter(prefix="/v1/models", tags=["Model Registry"])


@router.get(
    "",
    response_model=ModelsListResponse,
    dependencies=[Depends(require_roles([Role.VIEWER]))],
)
def list_models(request: Request) -> ModelsListResponse:
    """List registered models and active promoted version."""
    active = getattr(request.app.state, "active_model_version", "gemma-3-lora-v1")
    models = [
        ModelInfo(
            model_version=active,
            base_model="google/gemma-3-1b-it",
            stage="production",
            approved=True,
            created_at="2026-10-27T18:00:00Z",
            evaluation_summary={"mrr": 0.88, "groundedness": 4.6},
        ),
        ModelInfo(
            model_version="google/gemma-3-1b-it",
            base_model="google/gemma-3-1b-it",
            stage="baseline",
            approved=True,
            created_at="2026-10-01T00:00:00Z",
            evaluation_summary={"mrr": 0.72, "groundedness": 3.8},
        ),
    ]
    return ModelsListResponse(models=models, active_model=active)


@router.get(
    "/{version}",
    response_model=ModelInfo,
    dependencies=[Depends(require_roles([Role.VIEWER]))],
)
def get_model_details(version: str, request: Request) -> ModelInfo:
    """Retrieve details for a specific model version."""
    models_resp = list_models(request)
    for m in models_resp.models:
        if m.model_version == version:
            return m
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Model version '{version}' not found in registry.",
    )
