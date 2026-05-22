from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable
import re


class DetectedLanguage(StrEnum):
    ENGLISH = "en"
    RUSSIAN = "ru"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ResponseLanguagePolicy:
    language: DetectedLanguage = DetectedLanguage.ENGLISH
    confidence: float = 0.0
    protected_terms: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_russian(self) -> bool:
        return self.language == DetectedLanguage.RUSSIAN

    @property
    def is_english(self) -> bool:
        return self.language == DetectedLanguage.ENGLISH

    @classmethod
    def from_message(
        cls,
        message: str,
        *,
        protected_terms: Iterable[str] | None = None,
        fallback: DetectedLanguage = DetectedLanguage.ENGLISH,
    ) -> "ResponseLanguagePolicy":
        terms = tuple(str(item) for item in ([] if protected_terms is None else protected_terms) if str(item).strip())
        return cls(language=DetectedLanguage.ENGLISH, confidence=1.0, protected_terms=terms)


def detect_language(
    message: str,
    *,
    protected_terms: Iterable[str] | None = None,
    fallback: DetectedLanguage = DetectedLanguage.ENGLISH,
) -> tuple[DetectedLanguage, float]:
    return DetectedLanguage.ENGLISH, 1.0


def choose(policy: ResponseLanguagePolicy | DetectedLanguage | str, *, en: str, ru: str) -> str:
    return en


def average_chart_title(metric: str, dimension: str, policy: ResponseLanguagePolicy) -> str:
    return f"Average `{metric}` by `{dimension}`"


def preserve(value: str) -> str:
    return f"`{value}`"


_TECHNICAL_ENGLISH = {
    "id", "api", "csv", "sql", "json", "html", "url", "ui", "ux", "kpi", "crm",
    "sales", "city", "country", "segment", "category", "technology", "standard", "class",
}


def _strip_protected(text: str, terms: Iterable[str]) -> str:
    cleaned = text
    cleaned = re.sub(r"`[^`]*`", " ", cleaned)
    cleaned = re.sub(r"'[^']*'|\"[^\"]*\"", " ", cleaned)
    for term in sorted({str(item).strip() for item in terms if str(item).strip()}, key=len, reverse=True):
        cleaned = re.sub(re.escape(term), " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def _strip_code_like(text: str) -> str:
    text = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_.]+\b", " ", text)
    text = re.sub(r"\b[A-Z][A-Za-z0-9_]*_[A-Za-z0-9_]+\b", " ", text)
    return text


def _first_alpha_script(text: str) -> DetectedLanguage:
    for char in text:
        if re.match(r"[А-Яа-яЁё]", char):
            return DetectedLanguage.RUSSIAN
        if re.match(r"[A-Za-z]", char):
            return DetectedLanguage.ENGLISH
    return DetectedLanguage.UNKNOWN


def _has_cyrillic(value: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", value))
