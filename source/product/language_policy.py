from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable


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
