"""Web UI playground route serving an interactive developer chat interface."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["UI"])

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"


@router.get("/", response_class=HTMLResponse)
def playground_ui() -> HTMLResponse:
    """Serve the interactive Antigravity LLMOps Developer Playground."""
    if INDEX_HTML_PATH.exists():
        content = INDEX_HTML_PATH.read_text(encoding="utf-8")
    else:
        content = "<h1>Antigravity LLMOps</h1><p>UI template missing.</p>"
    return HTMLResponse(content=content)
