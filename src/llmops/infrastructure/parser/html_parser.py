"""HTML parser converting raw documentation snapshots into canonical Document objects."""

import hashlib
import re
import uuid
from typing import Any

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

from llmops.domain.models.document import Document, Section
from llmops.infrastructure.parser.normalizer import normalize_text

BOILERPLATE_TAGS = {
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "aside",
    "noscript",
    "svg",
    "form",
    "iframe",
}


class HTMLDocumentParser:
    """Parses raw HTML documentation into structured canonical Documents."""

    def __init__(self, parser_version: str = "1.0.0") -> None:
        self.parser_version = parser_version

    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from <title> tag or the first <h1>."""
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)

        title_tag = soup.find("title")
        if title_tag and title_tag.get_text(strip=True):
            # Clean common documentation title suffixes like " - FastAPI"
            raw_title = title_tag.get_text(strip=True)
            return re.split(r"\s+[—–-]\s+", raw_title)[0].strip()

        return "Untitled Document"

    def convert_code_blocks(self, container: Tag) -> None:
        """Transform <pre><code> blocks into markdown fences before text extraction."""
        for pre in container.find_all("pre"):
            code = pre.find("code")
            code_text = code.get_text() if code else pre.get_text()

            # Attempt to extract language class (e.g. language-python, highlight-python)
            lang = ""
            classes: list[str] = []
            code_classes = code.get("class") if code else None
            if isinstance(code_classes, list):
                classes.extend(str(c) for c in code_classes)
            elif isinstance(code_classes, str):
                classes.append(code_classes)

            pre_classes = pre.get("class")
            if isinstance(pre_classes, list):
                classes.extend(str(c) for c in pre_classes)
            elif isinstance(pre_classes, str):
                classes.append(pre_classes)

            for cls in classes:
                if cls.startswith("language-") or cls.startswith("highlight-"):
                    lang = cls.split("-", 1)[1]
                    break
                if cls in {"python", "py", "bash", "sh", "json", "yaml", "sql"}:
                    lang = cls
                    break

            fence = f"\n```{lang}\n{code_text.strip()}\n```\n"
            pre.replace_with(NavigableString(fence))

    def extract_sections(self, content_root: Tag) -> list[Section]:
        """Split content into structured Section objects based on heading tags."""
        sections: list[Section] = []
        current_heading = "Overview"
        current_level = 1
        current_content_parts: list[str] = []
        section_idx = 0

        # Traverse direct or nested elements
        heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}

        for element in content_root.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "pre", "blockquote", "table"]
        ):
            if element.name in heading_tags:
                # Flush previous section if it has content
                content_str = normalize_text("\n".join(current_content_parts))
                if content_str:
                    sec_hash = hashlib.sha256(content_str.encode("utf-8")).hexdigest()
                    sections.append(
                        Section(
                            section_id=f"sec-{section_idx:03d}",
                            heading=current_heading,
                            level=current_level,
                            content=content_str,
                            content_hash=sec_hash,
                        )
                    )
                    section_idx += 1
                    current_content_parts = []

                current_heading = element.get_text(strip=True) or "Untitled Section"
                current_level = int(element.name[1])
            else:
                text_block = element.get_text(separator=" ", strip=True)
                if text_block:
                    current_content_parts.append(text_block)

        # Flush final section
        content_str = normalize_text("\n".join(current_content_parts))
        if content_str:
            sec_hash = hashlib.sha256(content_str.encode("utf-8")).hexdigest()
            sections.append(
                Section(
                    section_id=f"sec-{section_idx:03d}",
                    heading=current_heading,
                    level=current_level,
                    content=content_str,
                    content_hash=sec_hash,
                )
            )

        return sections

    def parse(
        self,
        html_content: str,
        source_name: str,
        source_url: str,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Parse raw HTML content into a canonical Document."""
        soup = BeautifulSoup(html_content, "html.parser")

        # 1. Remove boilerplate elements
        for tag_name in BOILERPLATE_TAGS:
            for el in soup.find_all(tag_name):
                el.decompose()

        # 2. Extract title
        title = self.extract_title(soup)

        # 3. Locate content root (e.g. main, article, or body)
        content_root = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", {"role": "main"})
            or soup.find("div", class_=re.compile(r"md-content|document|content"))
            or soup.find("body")
            or soup
        )

        # 4. Convert code blocks into markdown fences
        if isinstance(content_root, Tag):
            self.convert_code_blocks(content_root)
            sections = self.extract_sections(content_root)
            raw_text = content_root.get_text(separator="\n", strip=True)
        else:
            sections = []
            raw_text = soup.get_text(separator="\n", strip=True)

        normalized_content = normalize_text(raw_text)
        doc_id = document_id or f"doc_{source_name}_{uuid.uuid4().hex[:8]}"
        content_hash = hashlib.sha256(normalized_content.encode("utf-8")).hexdigest()

        return Document(
            document_id=doc_id,
            source_name=source_name,
            source_url=source_url,
            title=title,
            version="1.0.0",
            content=normalized_content,
            content_hash=content_hash,
            parser_version=self.parser_version,
            metadata=metadata or {},
            sections=sections,
        )
