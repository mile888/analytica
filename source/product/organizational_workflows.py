from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from source.product.cross_investigation import CrossInvestigationPattern, build_analytical_playbooks
from source.product.investigation import ArtifactType, Finding, Investigation, ShareableReportTemplate


@dataclass(frozen=True)
class WorkflowExpectation:
    expectation_id: str
    title: str
    description: str
    evidence_types: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkflowStage:
    stage_id: str
    title: str
    description: str
    analysis_types: list[str] = field(default_factory=list)
    expectations: list[WorkflowExpectation] = field(default_factory=list)


@dataclass(frozen=True)
class WorkflowValidation:
    stage_id: str
    status: str
    message: str
    next_step: str = ""


@dataclass(frozen=True)
class OrganizationalWorkflow:
    workflow_id: str
    title: str
    description: str
    stages: list[WorkflowStage]
    recommended_for: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationExpectation:
    expectation_id: str
    title: str
    description: str
    severity: str = "info"
    applies_to: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    finding_id: str
    title: str
    status: str
    message: str
    severity: str = "info"


@dataclass(frozen=True)
class AnalyticalReviewRule:
    rule_id: str
    title: str
    analysis_types: list[str]
    expectation: str
    severity: str = "warning"


@dataclass(frozen=True)
class OrganizationalPlaybook:
    playbook_id: str
    title: str
    workflow_id: str
    analytical_sequence: list[str]
    chart_priorities: list[str]
    validation_questions: list[str]
    report_guidance: list[str]


@dataclass(frozen=True)
class ReviewFeedback:
    feedback_id: str
    title: str
    message: str
    severity: str = "info"
    related_finding_id: str | None = None


@dataclass(frozen=True)
class ValidationCheckpoint:
    checkpoint_id: str
    title: str
    status: str
    message: str
    severity: str = "info"


@dataclass(frozen=True)
class InvestigationReview:
    review_id: str
    status: str
    evidence_quality: str
    summary: str
    checkpoints: list[ValidationCheckpoint] = field(default_factory=list)
    feedback: list[ReviewFeedback] = field(default_factory=list)
    suggested_next_step: str = ""


@dataclass(frozen=True)
class ReportStandard:
    standard_id: str
    title: str
    required_sections: list[str]
    evidence_expectations: list[str]
    tone_guidance: str
    validation_guidance: list[str] = field(default_factory=list)


REAL_ANALYSIS_TYPES = {
    "grouped_metric",
    "outlier",
    "correlation",
    "trend",
    "distribution",
    "data_quality",
    "chart",
    "volume_relationship",
}


def default_organizational_workflows() -> list[OrganizationalWorkflow]:
    return [
        OrganizationalWorkflow(
            workflow_id="exploratory_investigation",
            title="Exploratory investigation",
            description="A lightweight path from dataset understanding to evidence-backed conclusions.",
            recommended_for=["grouped_metric", "distribution", "outlier", "trend"],
            stages=[
                _stage("dataset_understanding", "Dataset understanding", "Clarify records, metrics, dimensions and quality risks.", ["overview", "data_quality"]),
                _stage("group_comparison", "Group comparison", "Compare important metrics across meaningful dimensions.", ["grouped_metric", "distribution"]),
                _stage("outlier_detection", "Outlier detection", "Inspect unusual groups or records before drawing conclusions.", ["outlier"]),
                _stage("trend_analysis", "Trend analysis", "Check whether patterns remain stable over time when time fields exist.", ["trend"]),
                _stage("driver_analysis", "Driver analysis", "Test plausible drivers, confounders and segment effects.", ["correlation", "volume_relationship"]),
                _stage("validation_checks", "Validation checks", "Attach chart/table evidence and note limitations.", ["data_quality", "validation"]),
                _stage("executive_summary", "Executive summary", "Convert strongest conclusions into report-ready recommendations.", ["report"]),
            ],
        ),
        OrganizationalWorkflow(
            workflow_id="anomaly_investigation",
            title="Anomaly investigation",
            description="A focused path for unusual values, unstable segments and suspicious shifts.",
            recommended_for=["outlier", "data_quality"],
            stages=[
                _stage("locate_anomaly", "Locate anomaly", "Identify the metric, segment and magnitude of unusual behavior.", ["outlier"]),
                _stage("validate_records", "Validate records", "Check row-level support, sample size and data quality risk.", ["data_quality"]),
                _stage("explain_drivers", "Explain drivers", "Compare segment mix, volume and time concentration.", ["grouped_metric", "trend", "volume_relationship"]),
                _stage("document_risk", "Document risk", "Summarize uncertainty and recommended validation.", ["report"]),
            ],
        ),
        OrganizationalWorkflow(
            workflow_id="executive_reporting",
            title="Executive reporting",
            description="A concise path from strongest evidence to decision-ready narrative.",
            recommended_for=["report", "grouped_metric", "trend", "correlation"],
            stages=[
                _stage("select_conclusions", "Select conclusions", "Keep only the strongest report-ready findings.", ["grouped_metric", "outlier", "trend", "correlation"]),
                _stage("attach_evidence", "Attach evidence", "Support each conclusion with a chart, table or clear quantitative result.", ["chart", "table"]),
                _stage("state_uncertainty", "State uncertainty", "Name limitations, sample-size issues and non-causal relationships.", ["validation"]),
                _stage("recommend_action", "Recommend action", "Translate evidence into next analytical or business action.", ["report"]),
            ],
        ),
        OrganizationalWorkflow(
            workflow_id="data_quality_audit",
            title="Data quality audit",
            description="A review path for missingness, duplicates, inconsistent values and reliability risks.",
            recommended_for=["data_quality"],
            stages=[
                _stage("quality_scan", "Quality scan", "Measure missingness, duplicates, invalid values and sparse categories.", ["data_quality"]),
                _stage("impact_assessment", "Impact assessment", "Explain how quality issues affect analysis reliability.", ["data_quality"]),
                _stage("remediation_plan", "Remediation plan", "Recommend cleaning or validation steps before reporting.", ["validation"]),
            ],
        ),
    ]


def recommend_organizational_workflow(
    investigation: Investigation,
    patterns: list[CrossInvestigationPattern] | None = None,
) -> OrganizationalWorkflow:
    analysis_types = _analysis_types(investigation)
    pattern_types = {pattern.pattern_type for pattern in patterns or []}
    candidates = default_organizational_workflows()
    ranked = sorted(
        candidates,
        key=lambda workflow: len((analysis_types | pattern_types) & set(workflow.recommended_for)),
        reverse=True,
    )
    return ranked[0] if ranked else candidates[0]


def evaluate_workflow_progress(investigation: Investigation, workflow: OrganizationalWorkflow) -> list[WorkflowValidation]:
    analysis_types = _analysis_types(investigation)
    has_chart_or_table = _has_supporting_output(investigation)
    validations: list[WorkflowValidation] = []
    for stage in workflow.stages:
        matched = bool(analysis_types & set(stage.analysis_types))
        if stage.stage_id == "dataset_understanding" and investigation.linked_data_source_ids:
            matched = True
        if stage.stage_id in {"attach_evidence", "validation_checks"} and has_chart_or_table:
            matched = True
        status = "complete" if matched else "suggested"
        validations.append(
            WorkflowValidation(
                stage_id=stage.stage_id,
                status=status,
                message=f"{stage.title} is {'supported by the current investigation' if matched else 'a useful next checkpoint'}.",
                next_step="" if matched else stage.description,
            )
        )
    return validations


def validation_expectations_for_investigation(
    investigation: Investigation,
    patterns: list[CrossInvestigationPattern] | None = None,
) -> list[ValidationExpectation]:
    analysis_types = _analysis_types(investigation) | {pattern.pattern_type for pattern in patterns or []}
    expectations = [
        ValidationExpectation(
            expectation_id="chart_or_table_evidence",
            title="Attach visual or tabular evidence",
            description="Important conclusions should be backed by a chart, table or quantitative output.",
            applies_to=["grouped_metric", "outlier", "trend", "correlation"],
        )
    ]
    if "outlier" in analysis_types:
        expectations.append(
            ValidationExpectation(
                expectation_id="outlier_magnitude",
                title="State anomaly magnitude",
                description="Anomaly claims should include the size of the deviation and sample-size caution.",
                applies_to=["outlier"],
            )
        )
    if "correlation" in analysis_types:
        expectations.append(
            ValidationExpectation(
                expectation_id="correlation_caveat",
                title="Avoid causal overclaiming",
                description="Correlation findings should name possible shared drivers or segmentation effects.",
                applies_to=["correlation"],
            )
        )
    if "trend" in analysis_types:
        expectations.append(
            ValidationExpectation(
                expectation_id="trend_stability",
                title="Check trend stability",
                description="Trend claims should mention period counts, sparsity or seasonality where relevant.",
                applies_to=["trend"],
            )
        )
    if not investigation.findings:
        expectations.append(
            ValidationExpectation(
                expectation_id="generate_real_findings",
                title="Generate real analytical findings",
                description="Ask a focused comparison, trend, distribution or anomaly question before reporting.",
                severity="info",
                applies_to=["overview"],
            )
        )
    return expectations


def evidence_requirements_for_finding(finding: Finding) -> list[EvidenceRequirement]:
    metadata = finding.metadata or {}
    analysis_type = _metadata_text(metadata, "analysis_type")
    supporting_artifacts = _metadata_list(metadata, "supporting_artifact_ids") or list(finding.evidence_artifact_ids)
    evidence_strength = _metadata_text(metadata, "evidence_strength").lower()
    requirements: list[EvidenceRequirement] = []
    if analysis_type in {"grouped_metric", "outlier", "trend", "correlation", "distribution"}:
        status = "met" if supporting_artifacts or evidence_strength == "high" else "missing"
        requirements.append(
            EvidenceRequirement(
                requirement_id="supporting_output",
                finding_id=finding.finding_id,
                title="Supporting output",
                status=status,
                message="Finding has chart/table support." if status == "met" else "Attach a chart, table or quantitative output before treating this as report-ready.",
                severity="warning" if status == "missing" else "info",
            )
        )
    if analysis_type == "correlation":
        limitation = _metadata_text(metadata, "limitation").lower()
        has_caveat = any(token in limitation for token in ["caus", "association", "driver", "segment"])
        requirements.append(
            EvidenceRequirement(
                requirement_id="causal_caveat",
                finding_id=finding.finding_id,
                title="Causal caveat",
                status="met" if has_caveat else "missing",
                message="Correlation caveat is present." if has_caveat else "Add a caveat that association does not prove causality.",
                severity="warning" if not has_caveat else "info",
            )
        )
    if analysis_type == "outlier":
        validation = _metadata_text(metadata, "recommended_validation").lower()
        has_validation = any(token in validation for token in ["inspect", "raw", "sample", "record", "median"])
        requirements.append(
            EvidenceRequirement(
                requirement_id="outlier_validation",
                finding_id=finding.finding_id,
                title="Outlier validation",
                status="met" if has_validation else "missing",
                message="Outlier validation is explicit." if has_validation else "Add row-level or sample-size validation for this anomaly.",
                severity="warning" if not has_validation else "info",
            )
        )
    return requirements


def review_investigation(
    investigation: Investigation,
    patterns: list[CrossInvestigationPattern] | None = None,
) -> InvestigationReview:
    checkpoints: list[ValidationCheckpoint] = []
    feedback: list[ReviewFeedback] = []
    workflow = recommend_organizational_workflow(investigation, patterns)
    progress = evaluate_workflow_progress(investigation, workflow)
    completed = sum(1 for item in progress if item.status == "complete")
    checkpoints.append(
        ValidationCheckpoint(
            checkpoint_id="workflow_progress",
            title="Workflow progress",
            status="complete" if completed >= max(1, len(progress) // 2) else "suggested",
            message=f"{completed} of {len(progress)} suggested workflow checkpoints have evidence.",
        )
    )
    real_findings = [item for item in investigation.findings if _metadata_text(item.metadata or {}, "analysis_type") in REAL_ANALYSIS_TYPES]
    checkpoints.append(
        ValidationCheckpoint(
            checkpoint_id="real_findings",
            title="Analytical findings",
            status="complete" if real_findings else "suggested",
            message=f"{len(real_findings)} evidence-oriented findings are available." if real_findings else "No evidence-oriented findings yet.",
            severity="info" if real_findings else "warning",
        )
    )
    has_output = _has_supporting_output(investigation)
    checkpoints.append(
        ValidationCheckpoint(
            checkpoint_id="supporting_outputs",
            title="Supporting outputs",
            status="complete" if has_output else "suggested",
            message="Charts or tables are available as evidence." if has_output else "Add a chart or table for stronger evidence.",
            severity="info" if has_output else "warning",
        )
    )
    missing = []
    for finding in real_findings:
        for requirement in evidence_requirements_for_finding(finding):
            if requirement.status == "missing":
                missing.append(requirement)
                feedback.append(
                    ReviewFeedback(
                        feedback_id=f"{requirement.requirement_id}_{finding.finding_id}",
                        title=requirement.title,
                        message=requirement.message,
                        severity=requirement.severity,
                        related_finding_id=finding.finding_id,
                    )
                )
    evidence_quality = "strong" if has_output and not missing and real_findings else "developing" if real_findings else "early"
    status = "ready_for_report" if evidence_quality == "strong" else "needs_validation" if missing else "developing"
    next_step = _next_review_step(real_findings, has_output, missing, progress)
    return InvestigationReview(
        review_id=f"review_{investigation.investigation_id}",
        status=status,
        evidence_quality=evidence_quality,
        summary=_review_summary(status, evidence_quality),
        checkpoints=checkpoints,
        feedback=feedback[:8],
        suggested_next_step=next_step,
    )


def build_organizational_playbooks(
    patterns: list[CrossInvestigationPattern],
) -> list[OrganizationalPlaybook]:
    playbooks = []
    for playbook in build_analytical_playbooks(patterns):
        workflow_id = "exploratory_investigation"
        if "outlier" in playbook.pattern_types:
            workflow_id = "anomaly_investigation"
        elif "trend" in playbook.pattern_types:
            workflow_id = "exploratory_investigation"
        playbooks.append(
            OrganizationalPlaybook(
                playbook_id=f"org_{playbook.playbook_id}",
                title=playbook.title,
                workflow_id=workflow_id,
                analytical_sequence=playbook.common_flow,
                chart_priorities=playbook.common_charts,
                validation_questions=playbook.validation_questions,
                report_guidance=playbook.report_structure_hints,
            )
        )
    return playbooks or [
        OrganizationalPlaybook(
            playbook_id="org_exploratory_default",
            title="Exploratory investigation",
            workflow_id="exploratory_investigation",
            analytical_sequence=["understand dataset", "compare metric by dimension", "check outliers", "summarize evidence"],
            chart_priorities=["grouped bar", "distribution", "trend"],
            validation_questions=["Which conclusions have chart or table evidence?", "Could sparse groups distort the result?"],
            report_guidance=["Summarize strongest conclusions, evidence, limitations and recommended validations."],
        )
    ]


def report_standard_for_template(template: str | ShareableReportTemplate) -> ReportStandard:
    resolved = ShareableReportTemplate(template)
    if resolved == ShareableReportTemplate.PRODUCT_DECISION_MEMO:
        return ReportStandard(
            standard_id="product_decision_memo",
            title="Decision memo standard",
            required_sections=["Decision question", "Recommendation", "Supporting evidence", "Risks / limitations", "Next steps"],
            evidence_expectations=["Each recommendation should cite the strongest supporting finding or output."],
            tone_guidance="Decision-oriented, concise and explicit about uncertainty.",
            validation_guidance=["Separate evidence-backed conclusions from hypotheses.", "Name the validation needed before irreversible action."],
        )
    if resolved == ShareableReportTemplate.TECHNICAL_APPENDIX:
        return ReportStandard(
            standard_id="technical_appendix",
            title="Technical appendix standard",
            required_sections=["Technical artifacts", "Run trace"],
            evidence_expectations=["Preserve reproducibility details without turning them into executive conclusions."],
            tone_guidance="Precise and audit-friendly.",
            validation_guidance=["Keep technical details separate from user-facing conclusions."],
        )
    return ReportStandard(
        standard_id="executive_summary",
        title="Executive brief standard",
        required_sections=["Question", "Answer", "Key findings", "Evidence", "Limitations", "Next steps"],
        evidence_expectations=["Major conclusions should have chart, table or quantitative evidence."],
        tone_guidance="Senior analyst summary with evidence, caveats and recommended action.",
        validation_guidance=["Include limitations for sparse groups, outliers and correlations.", "Prioritize the strongest 3-5 conclusions."],
    )


def report_standard_hints(standard: ReportStandard, review: InvestigationReview | None = None) -> list[str]:
    hints = [
        *standard.evidence_expectations,
        *standard.validation_guidance,
    ]
    if review and review.suggested_next_step:
        hints.append(review.suggested_next_step)
    return _dedupe(hints)[:8]


def _stage(stage_id: str, title: str, description: str, analysis_types: list[str]) -> WorkflowStage:
    return WorkflowStage(
        stage_id=stage_id,
        title=title,
        description=description,
        analysis_types=analysis_types,
        expectations=[
            WorkflowExpectation(
                expectation_id=f"{stage_id}_evidence",
                title=f"{title} evidence",
                description=description,
                evidence_types=["chart", "table", "finding"],
            )
        ],
    )


def _analysis_types(investigation: Investigation) -> set[str]:
    values = set()
    for finding in investigation.findings:
        analysis_type = _metadata_text(finding.metadata or {}, "analysis_type")
        if analysis_type:
            values.add(analysis_type)
    for artifact in investigation.artifacts:
        artifact_type = getattr(artifact.artifact_type, "value", artifact.artifact_type)
        if artifact_type == ArtifactType.CHART.value:
            values.add("chart")
        elif artifact_type == ArtifactType.TABLE.value:
            values.add("table")
    return values


def _has_supporting_output(investigation: Investigation) -> bool:
    return any(
        getattr(artifact.artifact_type, "value", artifact.artifact_type) in {ArtifactType.CHART.value, ArtifactType.TABLE.value}
        for artifact in investigation.artifacts
    )


def _metadata_text(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    return str(value).strip() if isinstance(value, str) and value.strip() else ""


def _metadata_list(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata.get(key)
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _next_review_step(
    real_findings: list[Finding],
    has_output: bool,
    missing: list[EvidenceRequirement],
    progress: list[WorkflowValidation],
) -> str:
    if not real_findings:
        return "Ask a focused comparison, trend, distribution or anomaly question to create evidence-backed findings."
    if not has_output:
        return "Add a chart or table for the strongest finding before turning this into a report."
    if missing:
        return missing[0].message
    for item in progress:
        if item.status != "complete" and item.next_step:
            return item.next_step
    return "Convert the strongest evidence-backed conclusions into a concise report."


def _review_summary(status: str, evidence_quality: str) -> str:
    if status == "ready_for_report":
        return "The investigation has enough evidence structure for a draft report."
    if evidence_quality == "developing":
        return "The investigation has analytical findings, but some evidence or validation should be strengthened."
    return "The investigation is still early; generate focused findings before report review."


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value.strip())
    return result
