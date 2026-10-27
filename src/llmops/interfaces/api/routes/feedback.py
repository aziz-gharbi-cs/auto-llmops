"""User feedback collection endpoints for continuous evaluation (Section 26/45)."""

import uuid

from fastapi import APIRouter, Depends

from llmops.interfaces.api.schemas import FeedbackRequest, FeedbackResponse
from llmops.interfaces.api.security import Role, require_roles

router = APIRouter(prefix="/v1/feedback", tags=["Feedback & Monitoring"])


@router.post(
    "",
    response_model=FeedbackResponse,
    dependencies=[Depends(require_roles([Role.VIEWER]))],
)
def submit_feedback(request_body: FeedbackRequest) -> FeedbackResponse:
    """Submit user rating and qualitative feedback on generation answers."""
    feedback_id = f"fb-{uuid.uuid4().hex[:8]}"
    # In production, this persists to PostgreSQL or telemetry queue
    return FeedbackResponse(
        status="received",
        feedback_id=feedback_id,
    )
