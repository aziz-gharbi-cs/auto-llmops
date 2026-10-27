"""Unit tests for the text normalizer."""

from llmops.infrastructure.parser.normalizer import normalize_text


def test_normalize_line_endings_and_whitespace() -> None:
    """Verify normalize_text standardizes CRLF and trailing spaces."""
    raw = "Line 1   \r\nLine 2\t \r\n\r\n\r\n\r\nLine 3"
    result = normalize_text(raw)
    assert result == "Line 1\nLine 2\n\nLine 3"


def test_normalize_unescapes_html_entities() -> None:
    """Verify HTML entities are converted to plain characters."""
    raw = "FastAPI &amp; Pydantic &lt;version 2&gt; &quot;rocks&quot;"
    result = normalize_text(raw)
    assert result == 'FastAPI & Pydantic <version 2> "rocks"'


def test_normalize_strips_control_characters() -> None:
    """Verify non-printable control characters are stripped."""
    raw = "Text with \x00null and \x07bell"
    result = normalize_text(raw)
    assert result == "Text with null and bell"


def test_normalize_preserves_code_fences() -> None:
    """Verify markdown code blocks retain their structure and content."""
    raw = "```python\ndef hello():\n    return 'world'\n```"
    result = normalize_text(raw)
    assert result == raw
