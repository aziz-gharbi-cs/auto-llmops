"""Unit tests validating Argo CD GitOps manifests and CI/CD workflows (Milestone 12)."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ARGOCD_DIR = REPO_ROOT / "deployment" / "argocd"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


def test_argocd_applications_parse_and_validate() -> None:
    for env in ["dev", "staging", "prod"]:
        app_file = ARGOCD_DIR / f"application-{env}.yaml"
        assert app_file.exists(), f"Missing Argo CD application manifest: {app_file.name}"

        with app_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["apiVersion"] == "argoproj.io/v1alpha1"
        assert data["kind"] == "Application"
        assert data["metadata"]["name"] == f"llmops-{env}"
        assert data["spec"]["destination"]["namespace"] == f"llmops-{env}"

        # Verify automated sync policy
        sync_policy = data["spec"]["syncPolicy"]["automated"]
        assert sync_policy["prune"] is True
        assert sync_policy["selfHeal"] is True


def test_argocd_applicationset_manifest() -> None:
    set_file = ARGOCD_DIR / "applicationset.yaml"
    assert set_file.exists()

    with set_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["apiVersion"] == "argoproj.io/v1alpha1"
    assert data["kind"] == "ApplicationSet"
    assert "generators" in data["spec"]
    elements = data["spec"]["generators"][0]["list"]["elements"]
    environments = [el["env"] for el in elements]
    assert "dev" in environments
    assert "staging" in environments
    assert "prod" in environments


def test_github_actions_workflows_syntax_and_jobs() -> None:
    expected_workflows = [
        "ci.yml",
        "docker.yml",
        "data-validation.yml",
        "evaluation.yml",
        "training.yml",
        "release.yml",
    ]

    for wf_name in expected_workflows:
        wf_path = WORKFLOWS_DIR / wf_name
        assert wf_path.exists(), f"Missing workflow {wf_name}"

        with wf_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "name" in data
        assert "jobs" in data
        assert len(data["jobs"]) >= 1


def test_rollback_runbook_documentation() -> None:
    runbook_path = REPO_ROOT / "docs" / "runbooks" / "rollback.md"
    assert runbook_path.exists()
    content = runbook_path.read_text(encoding="utf-8")
    assert "argocd app rollback" in content
    assert "git revert" in content
    assert "ACTIVE_MODEL_VERSION" in content
