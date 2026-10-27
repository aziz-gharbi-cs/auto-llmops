# Cross-platform quality gate script for PowerShell
$ErrorActionPreference = "Stop"

Write-Host "=== 1. Checking formatting with Ruff ===" -ForegroundColor Cyan
python -m ruff format --check src tests
if ($LASTEXITCODE -ne 0) {
    Write-Error "Ruff format check failed. Run 'python -m ruff format src tests' to fix."
}

Write-Host "`n=== 2. Linting with Ruff ===" -ForegroundColor Cyan
python -m ruff check src tests
if ($LASTEXITCODE -ne 0) {
    Write-Error "Ruff linting failed."
}

Write-Host "`n=== 3. Static type checking with Mypy ===" -ForegroundColor Cyan
python -m mypy src tests
if ($LASTEXITCODE -ne 0) {
    Write-Error "Mypy type checking failed."
}

Write-Host "`n=== 4. Running unit tests with Pytest ===" -ForegroundColor Cyan
python -m pytest tests/unit -v --cov=src/llmops --cov-report=term-missing
if ($LASTEXITCODE -ne 0) {
    Write-Error "Unit tests failed."
}

Write-Host "`nAll quality gate checks passed successfully!" -ForegroundColor Green
