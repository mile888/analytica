from __future__ import annotations

from pathlib import Path
from typing import Any

from source.config import ARTIFACT_DIR
from source.product.investigation import Artifact, ArtifactType


REPORT_IMAGE_DIR = ARTIFACT_DIR / "report-images"


def artifact_report_snapshot(artifact: Artifact, *, image_path: str = "") -> dict[str, Any]:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    content = artifact.content if isinstance(artifact.content, dict) else {}
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_type": artifact.artifact_type.value,
        "chart_type": str(metadata.get("chart_type") or content.get("chart_type") or ""),
        "dataset_id": str(metadata.get("dataset_id") or ""),
        "dataset_ids": list(metadata.get("dataset_ids") or ([] if not metadata.get("dataset_id") else [metadata.get("dataset_id")])),
        "dataset_scope": str(metadata.get("dataset_scope") or ""),
        "title": artifact.title,
        "description": _artifact_caption(artifact),
        "image_path": image_path or str(metadata.get("image_path") or metadata.get("image_bytes_reference") or ""),
        "image_bytes_reference": image_path or str(metadata.get("image_bytes_reference") or metadata.get("image_path") or ""),
        "filters": metadata.get("filters") or content.get("filters") or [],
        "metric": metadata.get("metric") or content.get("metric") or "",
        "dimension": metadata.get("dimension") or content.get("dimension") or content.get("x") or "",
        "created_from_query": metadata.get("created_from_query") or "",
        "branch_id": metadata.get("branch_id") or "",
        "created_at": artifact.created_at.isoformat(),
        "content": artifact.content,
        "metadata": metadata,
    }


def ensure_artifact_report_image(artifact: Artifact) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    existing = str(metadata.get("image_path") or metadata.get("image_bytes_reference") or "")
    if existing and Path(existing).exists():
        return existing
    if artifact.artifact_type != ArtifactType.CHART:
        return ""
    REPORT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_IMAGE_DIR / f"{artifact.artifact_id}.png"
    if path.exists():
        return str(path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from source.product.exporter import _chart_artifact_figure

    fig = _chart_artifact_figure(plt, artifact)
    if fig is None:
        return ""
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def _artifact_caption(artifact: Artifact) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    bits = []
    metric = metadata.get("metric")
    dimension = metadata.get("dimension")
    filters = metadata.get("filters")
    query = metadata.get("created_from_query")
    if metric:
        bits.append(f"Metric: {metric}")
    if dimension:
        bits.append(f"Dimension: {dimension}")
    if filters:
        bits.append(f"Filters: {filters}")
    if query:
        bits.append(f"Originating question: {query}")
    return ". ".join(str(bit) for bit in bits if bit)
