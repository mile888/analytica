from __future__ import annotations

from typing import Any

from source.product.question_routing import QuestionIntentType, is_non_analytical_intent


ANALYTICAL_LEAKAGE_MARKERS = (
    " by `",
    " is led by ",
    " using average ",
    " using total ",
    "active comparison",
    "grouped",
    "metric",
    "dimension",
)


def non_analytical_output(question: str, intent: QuestionIntentType | str, *, has_dataset_context: bool = False) -> dict[str, Any]:
    text = non_analytical_response_text(question, intent, has_dataset_context=has_dataset_context)
    return {
        "summary": text,
        "final_answer": text,
        "structured_report": {
            "question": question,
            "summary": text,
            "key_findings": [],
            "evidence": ["Handled as a non-analytical conversational request."],
            "limitations": ["This is not a dataframe computation and should not be treated as a dataset finding."],
            "next_steps": [],
            "tool_timeline": [{"tool": "non_analytical_intent_escape", "status": "ok", "intent": str(intent)}],
        },
        "tool_timeline": [{"tool": "non_analytical_intent_escape", "status": "ok", "intent": str(intent)}],
        "trace_metadata": {
            "analysis_type": "non_analytical_conversation",
            "question_intent_type": str(intent),
            "suppress_key_findings": True,
            "disable_grouped_analysis_fallback": True,
        },
        "artifacts": [],
        "key_findings": [],
        "limitations": ["This is a conversational response, not a computed analytical result."],
        "critic_verdict": "",
    }


def non_analytical_response_text(question: str, intent: QuestionIntentType | str, *, has_dataset_context: bool = False) -> str:
    text = _norm(question)
    requested_non_english = _explicit_non_english_request(text)
    if _is_ambiguous_creative(text):
        if requested_non_english:
            return "Ты имеешь в виду творческое метафорическое сравнение или аналитическое сравнение по полям датасета? Я не буду запускать агрегацию, пока это неясно."
        return "Do you mean a creative metaphorical comparison, or an analytical comparison using dataset fields? I will not run a dataframe aggregation until that is clear."
    if str(intent) in {QuestionIntentType.CREATIVE_ANALOGY.value, QuestionIntentType.SEMANTIC_ANALOGY.value, "creative_analogy", "semantic_analogy"}:
        creative_entity = _extract_creative_entity(question)
        if requested_non_english:
            entity_label = creative_entity or "этому"
            return (
                f"Прямой аналитической связи с {entity_label} здесь нет, но как метафора сравнение возможно. "
                "В контексте расследования сильные группы или красивые средние значения могут выглядеть убедительно, "
                "пока не проверить устойчивость, выбросы и различия между сегментами. "
                "Это творческая интерпретация, а не бизнес-вывод из данных."
            )
        entity_label = creative_entity or "the reference"
        return (
            f"There is no direct analytical link to {entity_label}, but as a metaphor the comparison can work. "
            "Strong-looking groups or attractive averages can play a similar role until stability, outliers, and segment differences are checked. "
            "This is a creative interpretation, not a business conclusion from the data."
        )
    if str(intent) in {QuestionIntentType.META_PROJECT_QUESTION.value, "meta_project_question"}:
        if requested_non_english:
            return (
                "Это вопрос о работе агента, а не новая агрегация датасета. Я бы проверила следующий слой: правильно ли агент распознал намерение, не застрял ли в прошлой ветке, "
                "выбрал ли нужный датасет и не подменил ли вопрос старой метрикой. Для демо особенно полезно тестировать обычный анализ, follow-up, multi-dataset вопросы и такие escape-сценарии."
            )
        return (
            "This is a project/meta question, not a new dataframe aggregation. I would test whether the agent recognized the intent, avoided stale branch context, selected the right dataset, and did not replace the question with an old metric. "
            "For a demo, cover standard analysis, follow-ups, multi-dataset questions, and non-analytical escape cases."
        )
    if requested_non_english:
        return "Это не похоже на запрос к таблице, поэтому я отвечу разговорно и не буду продолжать старую аналитическую ветку."
    return "This does not look like a dataframe question, so I will answer conversationally rather than continuing the previous analytical branch."


def sanitize_non_analytical_text(text: str, question: str, intent: QuestionIntentType | str) -> str:
    if not is_non_analytical_intent(intent):
        return text
    normalized = _norm(text)
    if any(marker in normalized for marker in ANALYTICAL_LEAKAGE_MARKERS):
        return non_analytical_response_text(question, intent)
    return text


def _is_ambiguous_creative(text: str) -> bool:
    return any(marker in text for marker in ("compare this", "compare it", "сравни это", "сравнить это", "сравни с", "похоже на"))


def _explicit_non_english_request(text: str) -> bool:
    return any(marker in text for marker in ("ответь по русски", "на русском", "in russian", "answer in russian"))


def _extract_creative_entity(question: str) -> str:
    """Extract the named creative entity from the question."""
    text = _norm(question)
    import re
    for pattern in (
        r"(?:similarity|similarities|сходство)\s+(?:with|с)\s+(.+?)(?:\s*$|\s*[,.])",
        r"(?:compare|сравни)\s+(?:this|it|это)\s+(?:with|с|to)\s+(.+?)(?:\s*$|\s*[,.])",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    named_entities = {
        "la la land": "La La Land",
        "lalaland": "La La Land",
        "лалалэнд": "«Ла-Ла Лендом»",
        "лала лэнд": "«Ла-Ла Лендом»",
    }
    for key, label in named_entities.items():
        if key in text:
            return label
    return ""


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("_", " ").split())
