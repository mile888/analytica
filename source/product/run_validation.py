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
