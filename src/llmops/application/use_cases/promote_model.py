"""Use case for promoting models through quality gate to staging/production (Section 22)."""

import json
import logging
from pathlib import Path

import yaml

from llmops.domain.exceptions import QualityGateError
from llmops.domain.models.evaluation import EvaluationReport
from llmops.domain.models.training import ModelManifest

logger = logging.getLogger(__name__)


class PromoteModelUseCase:
    """Promotes candidate model to target stage only if quality gate criteria are verified."""

    def execute(
        self,
        model_version: str,
        target_stage: str = "staging",
        manifest_path: str | Path | None = None,
        evaluation_report_path: str | Path | None = None,
    ) -> ModelManifest:
        """Promote a candidate model after verifying Quality Gate approval."""
        if target_stage not in ("staging", "production"):
            raise QualityGateError(f"Invalid target promotion stage: '{target_stage}'")

        # 1. Locate Evaluation Report
        rep_path = (
            Path(evaluation_report_path)
            if evaluation_report_path
            else Path("artifacts/evaluation/evaluation_report.json")
        )
        if not rep_path.exists():
            raise QualityGateError(
                f"Cannot promote '{model_version}': evaluation report not found at '{rep_path}'. "
                "Execute 'llmops evaluate' prior to promotion (Section 22)."
            )

        report_data = json.loads(rep_path.read_text(encoding="utf-8"))
        report = EvaluationReport.model_validate(report_data)

        # 2. Verify Quality Gate status
        if not report.quality_gate_passed:
            reasons = "; ".join(report.rejection_reasons)
            raise QualityGateError(
                f"Cannot promote '{model_version}' to '{target_stage}': "
                f"Quality Gate rejected candidate. Reasons: {reasons}"
            )

        # 3. Locate Model Manifest
        man_path = (
            Path(manifest_path)
            if manifest_path
            else Path("artifacts/models/gemma-3-lora/adapter/model_manifest.yaml")
        )
        if not man_path.exists():
            raise QualityGateError(
                f"ModelManifest not found at '{man_path}'. Training artifact must exist."
            )

        manifest_data = yaml.safe_load(man_path.read_text(encoding="utf-8"))
        manifest = ModelManifest.model_validate(manifest_data)

        # 4. Update approval and attach evaluation results
        manifest.approved = True
        manifest.evaluation_report = report.model_dump(mode="json")

        # 5. Persist updated manifest
        manifest_dict = manifest.model_dump()
        manifest_dict["created_at"] = manifest.created_at.isoformat()
        manifest_dict["stage"] = target_stage
        man_path.write_text(yaml.safe_dump(manifest_dict), encoding="utf-8")

        logger.info(
            "Model '%s' successfully promoted to '%s' (Quality Gate passed).",
            model_version,
            target_stage,
        )
        return manifest
