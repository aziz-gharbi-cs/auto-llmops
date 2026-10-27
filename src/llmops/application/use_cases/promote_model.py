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
        model_version: str = "",
        target_stage: str = "staging",
        manifest_path: str | Path | None = None,
        evaluation_report_path: str | Path | None = None,
    ) -> ModelManifest:
        """Promote a candidate model after verifying Quality Gate approval."""
        if target_stage not in ("staging", "production"):
            raise QualityGateError(f"Invalid target promotion stage: '{target_stage}'")

        # 1. Verify explicit evaluation report path if specified
        rep_path: Path | None = None
        if evaluation_report_path:
            p = Path(evaluation_report_path)
            if not p.exists():
                raise QualityGateError(
                    f"Cannot promote '{model_version}': evaluation report not found at '{p}'. "
                    "Execute 'llmops evaluate' prior to promotion (Section 22)."
                )
            rep_path = p

        # 2. Locate Model Manifest (to extract model_version if omitted)
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
        resolved_version = model_version or manifest.model_version

        # 3. Locate Evaluation Report if not explicitly provided
        if rep_path is None:
            candidates = [
                Path("artifacts/evaluation/evaluation_report.json"),
                man_path.parent / "evaluation_report.json",
                man_path.parent.parent / "evaluation_report.json",
                man_path.parent.parent.parent / "evaluation_report.json",
            ]
            for cand in candidates:
                if cand.exists():
                    rep_path = cand
                    break

        if rep_path is None or not rep_path.exists():
            target_report = "artifacts/evaluation/evaluation_report.json"
            raise QualityGateError(
                f"Cannot promote '{resolved_version}': evaluation report not found at "
                f"'{target_report}'. Execute 'llmops evaluate' prior to promotion (Section 22)."
            )

        report_data = json.loads(rep_path.read_text(encoding="utf-8"))
        report = EvaluationReport.model_validate(report_data)

        # 3. Verify Quality Gate status
        if not report.quality_gate_passed:
            reasons = "; ".join(report.rejection_reasons)
            raise QualityGateError(
                f"Cannot promote '{resolved_version}' to '{target_stage}': "
                f"Quality Gate rejected candidate. Reasons: {reasons}"
            )

        # 4. Update approval and attach evaluation results
        manifest.approved = True
        manifest.stage = target_stage
        manifest.evaluation_report = report.model_dump(mode="json")

        # 5. Persist updated manifest
        manifest_dict = manifest.model_dump()
        manifest_dict["created_at"] = manifest.created_at.isoformat()
        manifest_dict["stage"] = target_stage
        man_path.write_text(yaml.safe_dump(manifest_dict), encoding="utf-8")

        logger.info(
            "Model '%s' successfully promoted to '%s' (Quality Gate passed).",
            resolved_version,
            target_stage,
        )
        return manifest
