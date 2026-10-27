"""Unit tests for the CLI interface."""

import pytest

from llmops.interfaces.cli import create_parser, main


def test_cli_parser_defaults() -> None:
    """Verify that the parser without args returns command as None."""
    parser = create_parser()
    args = parser.parse_args([])
    assert args.command is None


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --version flag prints the version."""
    parser = create_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "llmops 0.1.0" in captured.out


def test_cli_ingest_command() -> None:
    """Verify ingest subcommand arguments."""
    parser = create_parser()
    args = parser.parse_args(["ingest", "--config", "custom.yaml", "--source", "fastapi"])
    assert args.command == "ingest"
    assert args.config == "custom.yaml"
    assert args.source == "fastapi"


def test_cli_train_command() -> None:
    """Verify train subcommand arguments and profile choice."""
    parser = create_parser()
    args = parser.parse_args(["train", "--profile", "smoke"])
    assert args.command == "train"
    assert args.profile == "smoke"


def test_cli_promote_command() -> None:
    """Verify promote subcommand requires model-version."""
    parser = create_parser()
    args = parser.parse_args(
        ["promote", "--model-version", "v1.0.0", "--target-stage", "production"]
    )
    assert args.command == "promote"
    assert args.model_version == "v1.0.0"
    assert args.target_stage == "production"


def test_cli_main_execution(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify executing main with a command outputs acknowledgment."""
    exit_code = main(["ingest", "--source", "pydantic"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Executing subcommand 'ingest'" in captured.out
