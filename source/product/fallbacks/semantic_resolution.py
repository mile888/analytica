from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


SALES_LIKE_MARKERS = (
    "sales",
    "sale",
    "revenue",
    "amount",
    "total",
    "value",
    "price",
    "profit",
    "cost",
    "spend",
    "income",
    "volume",
    "quantity",
    "transactions",
    "order value",
    "transaction amount",
    "выруч",
    "продаж",
    "доход",
    "оборот",
)

PRODUCT_LIKE_MARKERS = ("product", "item", "sku", "товар", "продукт")
LOCATION_LIKE_MARKERS = ("city", "town", "location", "region", "state", "country", "site", "branch", "office", "territory", "zone", "area", "market", "district", "город", "регион", "страна")
CUSTOMER_LIKE_MARKERS = ("customer", "client", "account", "buyer", "user", "person", "member", "operator", "employee", "agent", "vendor", "supplier", "клиент")
ORDER_DATE_MARKERS = ("order date", "order_date", "orderdate", "date order", "ordered", "date", "time", "timestamp", "event date", "eventdate", "transaction date", "transactiondate", "created at", "createdat", "period", "month", "year", "дата заказа")
DELIVERY_DATE_MARKERS = ("ship date", "ship_date", "shipdate", "shipping date", "delivery date", "deliverydate", "delivered", "дата доставки")
SHIPPING_METHOD_MARKERS = ("ship mode", "shipping mode", "delivery mode", "shipping class", "delivery class", "channel")

GEO_SPECIFICITY = {
    "city": 90,
    "town": 88,
    "municipality": 86,
    "город": 90,
    "site": 70,
    "branch": 69,
    "office": 69,
    "location": 68,
    "territory": 67,
    "zone": 67,
    "area": 66,
    "district": 65,
    "market": 64,
    "state": 50,
    "province": 48,
    "region": 45,
    "регион": 45,
    "country": 25,
    "страна": 25,
}

HUMAN_ENTITY_MARKERS = ("name", "person", "people", "human", "client", "customer", "user", "buyer", "account", "member", "operator", "employee", "agent", "vendor", "supplier", "клиент", "имя")
IDENTIFIER_MARKERS = (" id", "_id", "-id", "uuid", "code", "key", "identifier", "number", "num")


@dataclass(frozen=True)
class ResolvedValue:
    column: str
    value: str
    source: str


def normalize(value: Any) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").casefold().split())


def column_by_markers(columns: list[str], markers: tuple[str, ...]) -> str | None:
    best: tuple[int, str] | None = None
    for column in columns:
        score = column_marker_score(column, markers)
        if score and (best is None or score > best[0]):
            best = (score, column)
    return best[1] if best else None


def column_marker_score(column: str, markers: tuple[str, ...]) -> int:
    normalized = normalize(column)
    return sum(20 + len(marker) for marker in markers if marker in normalized)


def resolve_dimension_column(question: str, columns: list[str], *, metric: str | None = None, df: pd.DataFrame | None = None) -> str | None:
    normalized = normalize(question)
    candidates = [column for column in columns if column != metric]
    for column in candidates:
        if normalize(column) in normalized:
            return column
    semantic_groups = (
        (("city", "cities", "город", "города", "городов", "городам", "городах"), LOCATION_LIKE_MARKERS, "geo"),
        (("customer", "customers", "client", "clients", "user", "person", "people", "клиент", "клиенты"), CUSTOMER_LIKE_MARKERS, "human"),
        (("product", "products", "item", "items", "товар", "продукт"), PRODUCT_LIKE_MARKERS, "product"),
        (("role", "roles", "profession", "professions", "job", "jobs", "occupation", "position", "професс", "должн", "роль"), ("job", "title", "role", "profession", "occupation", "position", "должн", "професс", "роль"), "human"),
        (("category", "categories", "group", "groups", "категор", "групп"), ("category", "категор", "class", "type", "group", "label"), "generic"),
        (("ship mode", "shipping mode", "достав"), ("ship mode", "shipping mode", "delivery mode"), "generic"),
        (("segment", "segments", "сегмент"), ("segment", "сегмент"), "generic"),
        (("bin", "bins"), ("bin", "bucket", "interval"), "generic"),
    )
    best: tuple[float, str] | None = None
    for question_markers, column_markers, family in semantic_groups:
        if not any(marker in normalized for marker in question_markers):
            continue
        for column in candidates:
            score = semantic_dimension_score(column, normalized, question_markers, column_markers, family, df=df)
            if score and (best is None or score > best[0]):
                best = (score, column)
    return best[1] if best else None


def semantic_dimension_score(
    column: str,
    normalized_question: str,
    question_markers: tuple[str, ...],
    column_markers: tuple[str, ...],
    family: str,
    *,
    df: pd.DataFrame | None = None,
) -> float:
    normalized_column = normalize(column)
    score = 0.0
    for marker in column_markers:
        if marker in normalized_column:
            score += 20 + len(marker)
    for marker in question_markers:
        if marker in normalized_question and marker in normalized_column:
            score += 35 + len(marker)
    if family == "geo":
        score += geo_specificity_score(normalized_column)
    if family == "human":
        score += human_readability_score(column, df[column] if df is not None and column in df.columns else None)
    if df is not None and column in df.columns:
        score += dimension_granularity_score(df[column], family)
    return score


def geo_specificity_score(normalized_column: str) -> int:
    return max((score for marker, score in GEO_SPECIFICITY.items() if marker in normalized_column), default=0)


def human_readability_score(column: str, series: pd.Series | None = None) -> int:
    normalized = normalize(column)
    score = 0
    if any(marker in normalized for marker in HUMAN_ENTITY_MARKERS):
        score += 40
    if "name" in normalized or "имя" in normalized:
        score += 50
    if _looks_identifier_name(normalized):
        score -= 80
    if series is not None:
        sample = [str(item).strip() for item in series.dropna().astype(str).head(80) if str(item).strip()]
        if sample:
            spaced = sum(1 for item in sample if " " in item)
            alphabetic = sum(1 for item in sample if any(char.isalpha() for char in item))
            code_like = sum(1 for item in sample if _looks_code_like_value(item))
            unique_ratio = series.nunique(dropna=True) / max(1, len(series.dropna()))
            if spaced / len(sample) >= 0.25:
                score += 30
            if alphabetic / len(sample) >= 0.8:
                score += 20
            if code_like / len(sample) >= 0.5:
                score -= 45
            if unique_ratio >= 0.5:
                score += 8
    return score


def dimension_granularity_score(series: pd.Series, family: str) -> float:
    if family != "geo":
        return 0.0
    unique = int(series.nunique(dropna=True))
    if unique <= 1:
        return -20.0
    return min(20.0, unique / 5.0)


def _looks_identifier_name(normalized: str) -> bool:
    if normalized == "id" or normalized.endswith(" id"):
        return True
    return any(marker in normalized for marker in IDENTIFIER_MARKERS)


def _looks_code_like_value(value: str) -> bool:
    compact = str(value or "").strip()
    if not compact:
        return False
    alnum = [char for char in compact if char.isalnum()]
    digits = sum(1 for char in alnum if char.isdigit())
    has_separator = any(char in compact for char in ("-", "_", "/", "."))
    return (has_separator and digits > 0) or (digits >= 3 and digits >= len(alnum) / 2)


def best_sales_like_metric(df: pd.DataFrame, question: str = "") -> str | None:
    numeric_columns = [str(column) for column in df.select_dtypes(include="number").columns]
    mentioned = mentioned_columns(numeric_columns, question)
    if mentioned:
        return mentioned[0]
    return column_by_markers(numeric_columns, SALES_LIKE_MARKERS)


def mentioned_columns(columns: list[str], question: str) -> list[str]:
    normalized_question = normalize(question)
    return [column for column in columns if normalize(column) in normalized_question]


def location_like_columns(df: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in df.columns]
    scored = [(column_marker_score(column, LOCATION_LIKE_MARKERS), column) for column in columns]
    return [column for score, column in sorted(scored, reverse=True) if score > 0]


def resolve_categorical_value(
    question: str,
    df: pd.DataFrame,
    *,
    preferred_columns: list[str] | None = None,
    exclude: set[str] | None = None,
) -> ResolvedValue | None:
    normalized_question = normalize(question)
    excluded = {item.casefold() for item in (exclude or set())}
    columns = preferred_columns or [str(column) for column in df.columns]
    for column in columns:
        if column not in df.columns or not _is_categorical_series(df[column]):
            continue
        exact = _match_observed_value(normalized_question, df[column], excluded)
        if exact:
            return ResolvedValue(column=column, value=exact, source="observed_value")
    return None


def _match_observed_value(normalized_question: str, series: pd.Series, excluded: set[str]) -> str:
    for value in series.dropna().astype(str).unique():
        value_norm = normalize(value)
        if value.casefold() not in excluded and len(value_norm) >= 3 and value_norm in normalized_question:
            return str(value)
    return ""


def _is_categorical_series(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype)


def _explicit_metric_column(df: pd.DataFrame, question: str) -> str | None:
    all_numeric_columns = [str(col) for col in df.select_dtypes(include="number").columns]
    mentioned = _mentioned_columns(all_numeric_columns, question)
    if mentioned:
        return mentioned[0]
    numeric_columns = [col for col in all_numeric_columns if not _looks_identifier_like(df[col], col)]
    mentioned = _mentioned_columns(numeric_columns, question)
    if mentioned:
        return mentioned[0]
    semantic = _semantic_column_match(
        numeric_columns,
        question,
        {
            "money": (
                "revenue", "sales", "sale", "salary", "price", "cost", "profit", "income", "pay", "compensation", "spend",
                "продаж", "продажи", "продажам", "выруч", "доход", "зарплат", "цена", "прибыл",
            ),
            "quality": ("rating", "score", "оцен", "рейтинг", "балл"),
        },
    )
    return semantic


def _explicit_dimension_column(df: pd.DataFrame, question: str, *, exclude: set[str]) -> str | None:
    candidates = _explicit_dimension_candidates(df, exclude=exclude)
    mentioned = _mentioned_columns(candidates, question)
    if mentioned:
        return mentioned[0]
    semantic_candidates = [col for col in candidates if not _looks_identifier_name_like(col)]
    return _semantic_column_match(semantic_candidates, question, _dimension_semantic_groups())


def _valid_column(df: pd.DataFrame, value: Any) -> str | None:
    name = str(value or "").strip()
    return name if name in set(str(col) for col in df.columns) else None


def _select_metric_column(df: pd.DataFrame, question: str, *, explicit_question: str | None = None) -> str | None:
    numeric_columns = [str(col) for col in df.select_dtypes(include="number").columns]
    numeric_columns = [col for col in numeric_columns if not _looks_identifier_like(df[col], col)]
    if not numeric_columns:
        return None
    metric_groups = {
        "metric": ("metric", "value", "amount", "score", "rating", "measure"),
        "money": (
            "revenue", "sales", "sale", "salary", "price", "cost", "profit", "income", "pay", "compensation", "spend",
            "продаж", "продажи", "продажам", "выруч", "доход", "зарплат", "цена", "прибыл"
        ),
        "volume": ("count", "quantity", "volume", "duration", "openings", "applicants", "users", "records", "колич", "число"),
    }
    explicit_text = explicit_question or question
    explicit_mentioned = _mentioned_columns(numeric_columns, explicit_text)
    if explicit_mentioned:
        return explicit_mentioned[0]
    explicit_semantic = _semantic_column_match(numeric_columns, explicit_text, metric_groups)
    if explicit_semantic:
        return explicit_semantic
    mentioned = _mentioned_columns(numeric_columns, question)
    if mentioned:
        return mentioned[0]
    semantic = _semantic_column_match(
        numeric_columns,
        question,
        metric_groups,
    )
    return semantic or numeric_columns[0]


def _select_dimension_column(
    df: pd.DataFrame,
    question: str,
    *,
    exclude: set[str],
    explicit_question: str | None = None,
) -> str | None:
    candidates = _dimension_candidates(df, exclude=exclude)

    if not candidates:
        return None
    dimension_groups = _dimension_semantic_groups()
    explicit_text = explicit_question or question
    explicit_mentioned = _mentioned_columns(candidates, explicit_text)
    if explicit_mentioned:
        return explicit_mentioned[0]
    explicit_semantic = _semantic_column_match(candidates, explicit_text, dimension_groups)
    if explicit_semantic:
        return explicit_semantic
    mentioned = _mentioned_columns(candidates, question)
    if mentioned:
        return mentioned[0]
    semantic = _semantic_column_match(candidates, question, dimension_groups)
    if semantic:
        return semantic
    return min(candidates, key=lambda col: int(df[col].nunique(dropna=True)))


def _dimension_candidates(df: pd.DataFrame, *, exclude: set[str]) -> list[str]:
    candidates = []
    for col in df.columns:
        name = str(col)
        if name in exclude:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        if _looks_timestamp_like(series):
            continue
        if _looks_long_text(series):
            continue
        unique_count = int(series.nunique(dropna=True))
        if 2 <= unique_count <= max(10, min(80, int(len(df) * 0.75))) and not _looks_identifier_name_like(name):
            candidates.append(name)
    return candidates


def _is_meaningful_dimension(df: pd.DataFrame, dimension_col: str | None) -> bool:
    if not dimension_col or dimension_col not in df.columns:
        return False
    if _looks_identifier_name_like(dimension_col):
        return False
    series = df[dimension_col]
    if _looks_timestamp_like(series) or _looks_long_text(series):
        return False
    unique_count = int(series.nunique(dropna=True))
    if unique_count <= 1:
        return False
    if len(df) and unique_count / max(len(df), 1) > 0.9:
        return False
    normalized = _normalize(dimension_col)
    if any(marker in normalized for marker in ("postal", "zip", "postcode")):
        return False
    return True


def _dimension_explicitly_requested(question: str, dimension_col: str | None) -> bool:
    if not dimension_col:
        return False
    normalized_question = _normalize(question)
    normalized_dimension = _normalize(dimension_col)
    if normalized_dimension in normalized_question:
        return True
    return bool(_explicit_entity_priority(normalized_question, normalized_dimension))


def _explicit_dimension_candidates(df: pd.DataFrame, *, exclude: set[str]) -> list[str]:
    """Candidate dimensions for explicit user entities.

    Keep explicit user-requested dimensions even when they have many unique values.
    """

    candidates = []
    for col in df.columns:
        name = str(col)
        if name in exclude:
            continue
        series = df[col]
        if _looks_timestamp_like(series):
            continue
        if _looks_long_text(series):
            continue
        if int(series.nunique(dropna=True)) >= 1:
            candidates.append(name)
    return candidates


def _dimension_semantic_groups() -> dict[str, tuple[str, ...]]:
    return {
        "role": (
            "role", "roles", "profession", "professions", "job", "jobs", "title", "occupation", "position",
            "професс", "профессии", "профессиям", "должн", "роль", "ролям",
        ),
        "category": (
            "category", "categories", "segment", "segments", "type", "types", "class", "group", "groups",
            "категор", "сегмент", "сегменты", "сегментам", "тип", "типы", "групп",
        ),
        "organization": (
            "company", "vendor", "customer", "client", "product", "industry", "department", "team",
            "компан", "клиент", "продукт", "индустр", "отрасл",
        ),
        "place": (
            "region", "city", "country", "location", "market", "area",
            "город", "города", "городам", "городах", "городов", "городе",
            "регион", "регионам", "регионах", "страна", "странам", "локац",
        ),
    }


def _select_timestamp_column(df: pd.DataFrame, question: str) -> str | None:
    candidates = [str(col) for col in df.columns if _looks_timestamp_like(df[col])]
    if not candidates:
        return None
    mentioned = _mentioned_columns(candidates, question)
    return mentioned[0] if mentioned else candidates[0]


def _has_correlation_candidate(df: pd.DataFrame, metric_col: str) -> bool:
    numeric_columns = [str(col) for col in df.select_dtypes(include="number").columns]
    candidates = [col for col in numeric_columns if col != metric_col and not _looks_identifier_like(df[col], col)]
    return bool(candidates)


def _mentioned_columns(columns: list[str], question: str) -> list[str]:
    normalized_question = _normalize(question)
    return [col for col in columns if _normalize(col) in normalized_question]


def _semantic_column_by_markers(df: pd.DataFrame, markers: tuple[str, ...]) -> str | None:
    return column_by_markers([str(column) for column in df.columns], markers)


def _column_name_matches(column: str, markers: tuple[str, ...]) -> bool:
    return _column_marker_score(column, markers) > 0


def _column_marker_score(column: str, markers: tuple[str, ...]) -> int:
    normalized = _normalize(column)
    return sum(20 + len(marker) for marker in markers if marker in normalized)


def _semantic_column_match(columns: list[str], question: str, groups: dict[str, tuple[str, ...]]) -> str | None:
    normalized_question = _normalize(question)
    question_tokens = set(normalized_question.split())
    best: tuple[float, str] | None = None
    for column in columns:
        normalized_column = _normalize(column)
        column_tokens = set(normalized_column.split())
        score = float(len(column_tokens & question_tokens) * 3)
        score += _explicit_entity_priority(normalized_question, normalized_column)
        for group_name, group_tokens in groups.items():
            group_set = set(group_tokens)
            for marker in group_tokens:
                if marker in normalized_question and marker in normalized_column:
                    score += 24
            if question_tokens & group_set and column_tokens & group_set:
                score += 5
            if group_name == "place" and question_tokens & group_set:
                score += geo_specificity_score(normalized_column)
            if group_name == "organization" and question_tokens & group_set:
                score += human_readability_score(column)
        if score and (best is None or score > best[0]):
            best = (score, column)
    return best[1] if best else None


def _explicit_entity_priority(question: str, column_name: str) -> int:
    pairs = (
        (("город", "города", "городов", "городам", "городах", "city", "cities"), ("city", "город")),
        (("страна", "страны", "country", "countries"), ("country", "страна")),
        (("регион", "регионы", "region", "regions"), ("region", "регион")),
        (("сегмент", "сегменты", "segment", "segments"), ("segment", "сегмент")),
        (("продукт", "товар", "product", "products"), ("product", "продукт", "товар")),
        (("професс", "должн", "роль", "job title", "role", "profession"), ("job", "title", "role", "profession", "должн", "професс")),
    )
    for question_markers, column_markers in pairs:
        if any(marker in question for marker in question_markers) and any(marker in column_name for marker in column_markers):
            return 60
    return 0


def _missing_explicit_entity(question: str, df: pd.DataFrame) -> str | None:
    normalized = _normalize(question)
    entity_markers = {
        "city": (("город", "города", "городов", "городам", "городах", "city", "cities"), ("city", "город")),
        "country": (("страна", "страны", "country", "countries"), ("country", "страна")),
        "region": (("регион", "регионы", "region", "regions"), ("region", "регион")),
    }
    column_names = [_normalize(str(column)) for column in df.columns]
    for entity, (question_markers, column_markers) in entity_markers.items():
        if not any(marker in normalized for marker in question_markers):
            continue
        if not any(any(marker in column for marker in column_markers) for column in column_names):
            return entity
    return None


def _missing_explicit_metric_request(question: str, column_names: list[str]) -> str | None:
    normalized = _normalize(question)
    requested = _requested_metric_label(normalized)
    if not requested:
        return None
    normalized_columns = [_normalize(column) for column in column_names]
    if requested in {"revenue", "sales", "sale", "выручка", "выручку", "доход", "продажи", "продаж", "оборот"}:
        if any(any(marker in column for marker in SALES_LIKE_MARKERS) for column in normalized_columns):
            return None
    marker_groups = {
        "sales-like metric": ("sales", "sale", "продаж", "продажи"),
        "revenue-like metric": ("revenue", "выруч"),
        "profit-like metric": ("profit", "прибыл"),
        "salary-like metric": ("salary", "зарплат", "compensation", "pay"),
        "cost-like metric": ("cost", "expense", "затрат", "расход"),
        "price-like metric": ("price", "цена"),
        "rating-like metric": ("rating", "рейтинг"),
        "score-like metric": ("score", "оцен", "балл"),
        "applicant-count metric": ("applicant", "applicants", "кандидат", "заяв"),
        "opening-count metric": ("opening", "openings", "ваканс"),
    }
    markers = marker_groups.get(requested, (_normalize(requested),))
    if any(any(marker in column for marker in markers) for column in normalized_columns):
        return None
    return requested


def _requested_metric_label(normalized_question: str) -> str | None:
    checks = (
        ("sales", "sale", "продаж", "продажи"),
        ("revenue", "выруч"),
        ("profit", "прибыл"),
        ("salary", "зарплат", "compensation", "pay"),
        ("cost", "expense", "затрат", "расход"),
        ("price", "цена"),
        ("rating", "рейтинг"),
        ("score", "оцен", "балл"),
        ("applicant", "applicants", "кандидат", "заяв"),
        ("opening", "openings", "ваканс"),
    )
    for markers in checks:
        match = _matched_question_word(normalized_question, markers)
        if match:
            return match
    return None


def _matched_question_word(normalized_question: str, markers: tuple[str, ...]) -> str | None:
    for word in normalized_question.split():
        if any(marker in word for marker in markers):
            return word.strip("`'\".,:;!?()[]{}")
    return None


def _normalize(value: str) -> str:
    return " ".join(str(value).replace("_", " ").replace("-", " ").lower().split())


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _looks_identifier_like(series: pd.Series, name: str) -> bool:
    lowered = name.lower()
    if _looks_identifier_name_like(name):
        return True
    non_null = series.dropna()
    return len(non_null) > 20 and non_null.nunique(dropna=True) / max(len(non_null), 1) > 0.9


def _looks_identifier_name_like(name: str) -> bool:
    lowered = name.lower()
    if lowered == "id" or lowered.endswith(" id") or lowered.endswith("_id") or lowered.endswith("-id"):
        return True
    return "code" in lowered or "postal" in lowered


def _looks_timestamp_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = series.dropna().head(50)
    if sample.empty:
        return False
    text_sample = sample.astype(str)
    if not text_sample.str.contains(r"[-/:T]", regex=True).any():
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.7)


def _looks_long_text(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(30)
    if sample.empty:
        return False
    return bool(sample.str.len().mean() > 80)
