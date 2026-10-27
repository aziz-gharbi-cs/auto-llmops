"""Unit tests validating Terraform IaC architecture and security (Milestone 13)."""

from pathlib import Path

TERRAFORM_ROOT = Path(__file__).resolve().parents[2] / "deployment" / "terraform"


def test_terraform_modules_structure() -> None:
    modules_dir = TERRAFORM_ROOT / "modules"
    assert modules_dir.is_dir()

    for mod_name in ["networking", "storage", "kubernetes"]:
        mod_path = modules_dir / mod_name
        assert mod_path.is_dir(), f"Missing Terraform module: {mod_name}"
        assert (mod_path / "main.tf").exists(), f"Missing main.tf in module {mod_name}"
        assert (mod_path / "variables.tf").exists(), f"Missing variables.tf in module {mod_name}"
        assert (mod_path / "outputs.tf").exists(), f"Missing outputs.tf in module {mod_name}"


def test_terraform_environments_structure() -> None:
    envs_dir = TERRAFORM_ROOT / "environments"
    assert envs_dir.is_dir()

    for env_name in ["dev", "prod"]:
        env_path = envs_dir / env_name
        assert env_path.is_dir(), f"Missing environment: {env_name}"
        assert (env_path / "main.tf").exists(), f"Missing main.tf in environment {env_name}"
        assert (env_path / "variables.tf").exists()
        assert (env_path / "terraform.tfvars.example").exists()


def test_no_hardcoded_secrets_in_terraform() -> None:
    forbidden_tokens = [
        "AKIA",  # AWS Access Key ID prefix
        "aws_secret_access_key",
        "BEGIN RSA PRIVATE KEY",
        "BEGIN OPENSSH PRIVATE KEY",
    ]

    for tf_file in TERRAFORM_ROOT.rglob("*.tf*"):
        content = tf_file.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            rel_path = tf_file.relative_to(TERRAFORM_ROOT)
            assert token not in content, f"Forbidden security token '{token}' in {rel_path}"


def test_kubernetes_module_gpu_tolerations_and_taints() -> None:
    k8s_main = TERRAFORM_ROOT / "modules" / "kubernetes" / "main.tf"
    content = k8s_main.read_text(encoding="utf-8")

    assert "gpu_nodes" in content
    assert "nvidia.com/gpu" in content
    assert "NO_SCHEDULE" in content
    assert "accelerator" in content


def test_storage_module_encryption_and_versioning() -> None:
    storage_main = TERRAFORM_ROOT / "modules" / "storage" / "main.tf"
    content = storage_main.read_text(encoding="utf-8")

    assert "aws_s3_bucket_versioning" in content
    assert "Enabled" in content
    assert "aws_s3_bucket_server_side_encryption_configuration" in content
    assert "AES256" in content
    assert "aws_s3_bucket_public_access_block" in content
