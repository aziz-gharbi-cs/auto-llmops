"""Unit tests for the HTML document parser."""

from llmops.infrastructure.parser.html_parser import HTMLDocumentParser


def test_parser_strips_boilerplate_tags() -> None:
    """Verify script, style, nav, and footer tags are removed."""
    html_content = """
    <html>
        <head>
            <title>FastAPI Tutorial</title>
            <style>body { font-size: 14px; }</style>
            <script>console.log("tracker");</script>
        </head>
        <body>
            <nav><a href="/">Home</a><a href="/about">About</a></nav>
            <main>
                <h1>First Steps</h1>
                <p>FastAPI is very fast.</p>
            </main>
            <footer>Copyright 2026</footer>
        </body>
    </html>
    """
    parser = HTMLDocumentParser()
    doc = parser.parse(
        html_content=html_content,
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/tutorial/",
    )

    assert "First Steps" in doc.title
    assert "FastAPI is very fast." in doc.content
    assert "console.log" not in doc.content
    assert "font-size" not in doc.content
    assert "Copyright 2026" not in doc.content


def test_parser_converts_pre_code_to_markdown_fences() -> None:
    """Verify <pre><code class="language-python"> becomes markdown code block."""
    html_content = """
    <html>
        <body>
            <h1>Code Example</h1>
            <pre><code class="language-python">from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}</code></pre>
        </body>
    </html>
    """
    parser = HTMLDocumentParser()
    doc = parser.parse(
        html_content=html_content,
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/example/",
    )

    assert "```python" in doc.content
    assert 'return {"Hello": "World"}' in doc.content
    assert "```" in doc.content


def test_parser_extracts_structured_sections() -> None:
    """Verify parser segments content into Section models with headings and levels."""
    html_content = """
    <html>
        <body>
            <h1>Main Title</h1>
            <p>Main introduction paragraph.</p>
            <h2>Installation</h2>
            <p>Run pip install fastapi.</p>
            <h2>Usage</h2>
            <p>Import and instantiate FastAPI.</p>
        </body>
    </html>
    """
    parser = HTMLDocumentParser()
    doc = parser.parse(
        html_content=html_content,
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/",
    )

    assert len(doc.sections) == 3
    assert doc.sections[0].heading == "Main Title"
    assert doc.sections[0].level == 1
    assert "Main introduction paragraph." in doc.sections[0].content

    assert doc.sections[1].heading == "Installation"
    assert doc.sections[1].level == 2
    assert "Run pip install fastapi." in doc.sections[1].content

    assert doc.sections[2].heading == "Usage"
    assert doc.sections[2].level == 2
