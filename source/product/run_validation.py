from __future__ import annotations

from typing import Any

from source.product.investigation import ArtifactVisibility, Investigation


def validate_investigation_result(investigation: Investigation, run_id: str | None = None) -> dict[str, Any]:
    artifacts = [
        artifact
        for artifact in investigation.artifacts
        if run_id is None or artifact.run_id == run_id
    ]
    findings = [
        finding
        for finding in investigation.findings
        if run_id is None or finding.run_id == run_id
    ]
    report = investigation.report if investigation.report and (run_id is None or investigation.report.run_id == run_id) else None
    warnings: list[str] = []

    if not findings:
        warnings.append("No findings were generated.")
    if not artifacts:
        warnings.append("No artifacts were generated.")
    if artifacts and all(artifact.visibility != ArtifactVisibility.USER for artifact in artifacts):
        warnings.append("Only technical artifacts were generated.")
    if not report:
        warnings.append("No DecisionReport was generated.")
    if not findings and not artifacts and not report:
        warnings.append("Empty investigation result.")
    trace = report.metadata.get("trace_metadata") if report and isinstance(report.metadata, dict) and isinstance(report.metadata.get("trace_metadata"), dict) else {}
    dataset_scope = str(trace.get("dataset_scope") or "")
    dataset_ids = trace.get("dataset_ids") if isinstance(trace.get("dataset_ids"), list) else []
    branch_scopes = trace.get("dataset_execution_scopes") if isinstance(trace.get("dataset_execution_scopes"), list) else []
    if dataset_scope == "cross_dataset":
        if len(dataset_ids) >= 2 and len(branch_scopes) < 2:
            warnings.append("Cross-dataset question collapsed before branch-scoped execution.")
        chart_artifacts = [artifact for artifact in artifacts if str(getattr(getattr(artifact, "artifact_type", None), "value", getattr(artifact, "artifact_type", None))).lower() == "chart"]
        if str(trace.get("operation") or "") == "PARALLEL_VISUAL_ANALYSIS" and len(chart_artifacts) < len(dataset_ids):
            warnings.append("Parallel visual analysis produced fewer chart artifacts than selected datasets.")
        artifact_dataset_ids = {
            str((getattr(artifact, "metadata", {}) or {}).get("dataset_id") or "")
            for artifact in artifacts
        }
        if len(dataset_ids) >= 2 and len({item for item in artifact_dataset_ids if item}) == 1 and len(chart_artifacts) <= 1:
            warnings.append("Cross-dataset artifacts appear to reference only one dataset.")

    latest_run = investigation.runs[-1] if investigation.runs else None
    execution_error = bool(latest_run and latest_run.error)

    return {
        "findings_count": len(findings),
        "artifacts_count": len(artifacts),
        "reports_generated": bool(report),
        "execution_error_present": execution_error,
        "technical_artifacts_only": bool(artifacts) and all(
            artifact.visibility != ArtifactVisibility.USER for artifact in artifacts
        ),
        "empty_result_warning": not findings and not artifacts and not report,
        "warnings": warnings,
    }
