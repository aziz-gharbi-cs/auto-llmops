"""Unit tests validating Kubernetes manifests and Helm chart specifications (Milestone 11)."""

from pathlib import Path

import yaml

HELM_ROOT = Path(__file__).resolve().parents[2] / "deployment" / "helm" / "llmops"


def test_chart_metadata() -> None:
    chart_file = HELM_ROOT / "Chart.yaml"
    assert chart_file.exists()

    with chart_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["apiVersion"] == "v2"
    assert data["name"] == "llmops"
    assert "version" in data
    assert "appVersion" in data


def test_environment_values_files_exist_and_parse() -> None:
    for env in ["dev", "staging", "prod"]:
        val_file = HELM_ROOT / f"values-{env}.yaml"
        assert val_file.exists(), f"Missing values-{env}.yaml"

        with val_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["global"]["environment"] in ["development", "staging", "production"]
        assert "api" in data
        assert "resources" in data["api"] or "replicaCount" in data["api"]


def test_values_prod_autoscaling_and_resilience() -> None:
    prod_file = HELM_ROOT / "values-prod.yaml"
    with prod_file.open("r", encoding="utf-8") as f:
        prod_data = yaml.safe_load(f)

    api_conf = prod_data["api"]
    assert api_conf["replicaCount"] >= 3
    assert api_conf["autoscaling"]["enabled"] is True
    assert api_conf["autoscaling"]["minReplicas"] >= 3
    assert api_conf["autoscaling"]["maxReplicas"] >= 10
    assert api_conf["podDisruptionBudget"]["enabled"] is True
    assert api_conf["podDisruptionBudget"]["minAvailable"] >= 2


def test_helm_template_files_structure() -> None:
    templates_dir = HELM_ROOT / "templates"
    assert templates_dir.is_dir()

    expected_templates = [
        "_helpers.tpl",
        "configmap.yaml",
        "secret.yaml",
        "service-api.yaml",
        "service-inference.yaml",
        "deployment-api.yaml",
        "deployment-inference.yaml",
        "ingress.yaml",
        "hpa-api.yaml",
        "pdb-api.yaml",
    ]

    for t_name in expected_templates:
        t_path = templates_dir / t_name
        assert t_path.exists(), f"Expected Helm template {t_name} does not exist"
        content = t_path.read_text(encoding="utf-8")
        assert len(content.strip()) > 0


def test_api_deployment_spec_contains_probes_and_security() -> None:
    deploy_file = HELM_ROOT / "templates" / "deployment-api.yaml"
    content = deploy_file.read_text(encoding="utf-8")

    # Verify key resilience and security directives
    assert "livenessProbe" in content
    assert "readinessProbe" in content
    assert "/health" in content or "livenessProbe.httpGet.path" in content
    assert "/ready" in content or "readinessProbe.httpGet.path" in content
    assert "runAsNonRoot" in content
    assert "runAsUser" in content
    assert "configMapRef" in content
    assert "secretRef" in content


def test_inference_deployment_spec_contains_gpu_tolerations() -> None:
    inf_file = HELM_ROOT / "templates" / "deployment-inference.yaml"
    content = inf_file.read_text(encoding="utf-8")

    assert "nodeSelector" in content
    assert "tolerations" in content

    # Check values.yaml defines GPU nodeSelector and tolerations
    values_file = HELM_ROOT / "values.yaml"
    with values_file.open("r", encoding="utf-8") as f:
        values_data = yaml.safe_load(f)

    inf_val = values_data["inference"]
    assert "accelerator" in inf_val["nodeSelector"]
    assert any("nvidia.com/gpu" in t.get("key", "") for t in inf_val["tolerations"])
