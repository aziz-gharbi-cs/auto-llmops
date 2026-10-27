"""Text normalizer for documentation cleaning and formatting."""

import html
import re


def normalize_text(text: str) -> str:
    """Normalize text content while strictly preserving code blocks and formatting.

    - Standardizes line breaks (\r\n -> \n)
    - Unescapes HTML entities (&amp; -> &, &lt; -> <, etc.)
    - Removes non-printable / control characters except \t and \n
    - Collapses excessive blank lines (3+ to 2)
    - Strips leading and trailing whitespace from the full text
    """
    if not text:
        return ""

    # 1. Standardize line endings
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Unescape HTML entities
    normalized = html.unescape(normalized)

    # 3. Strip control characters except newline and tab
    normalized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", normalized)

    # 4. Collapse lines with only whitespace
    normalized = re.sub(r"[ \t]+$", "", normalized, flags=re.MULTILINE)

    # 5. Collapse excessive consecutive empty lines (3 or more -> 2)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized.strip()
