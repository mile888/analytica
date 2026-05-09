"""
Engine factory — encapsulates data loading and processing for each backend.

Usage:
    from source.engine import create_engine
    engine = create_engine("pandas")
    df = engine.read_csv("path/to/file.csv")
    schema = engine.schema_text(df)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from source.llm.llm_config import EngineName


# ═══════════════════════════════════════════════════════════════════════
#  Base class
# ═══════════════════════════════════════════════════════════════════════

class BaseEngine(ABC):
    """Abstract compute engine with two responsibilities:
    1. **Data loading** — read from files (CSV, Parquet, Excel) or databases (SQL).
    2. **Data processing** — convert, introspect schema, build sandbox env.
    """

    name: EngineName

    # ── Data loading ────────────────────────────────────────────────

    @abstractmethod
    def read_csv(self, path: str, **kwargs: Any) -> Any:
        """Read a CSV file into the engine's native table format."""

    @abstractmethod
    def read_parquet(self, path: str, **kwargs: Any) -> Any:
        """Read a Parquet file into the engine's native table format."""

    @abstractmethod
    def read_excel(self, path: str, **kwargs: Any) -> Any:
        """Read an Excel file into the engine's native table format."""

    @abstractmethod
    def read_sql(self, query: str, connection: Any, **kwargs: Any) -> Any:
        """Execute a SQL query and return results in the engine's native format."""

    # ── Data processing ─────────────────────────────────────────────

    @abstractmethod
    def ensure_table(self, df: Any) -> Any:
        """Convert any supported table type into this engine's native format."""

    @abstractmethod
    def to_pandas(self, df: Any, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Convert the engine's native table back to a pandas DataFrame."""

    @abstractmethod
    def schema_text(self, df: Any) -> str:
        """Return a textual schema description for LLM prompts."""

    @abstractmethod
    def exec_env(self, df: Any) -> Dict[str, Any]:
        """Build the sandbox environment dict for safe_exec."""


# ═══════════════════════════════════════════════════════════════════════
#  Pandas
# ═══════════════════════════════════════════════════════════════════════

class PandasEngine(BaseEngine):
    name: EngineName = "pandas"

    # ── Loading ─────────────────────────────────────────────────────

    def read_csv(self, path: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_csv(path, **kwargs)

    def read_parquet(self, path: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_parquet(path, **kwargs)

    def read_excel(self, path: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_excel(path, **kwargs)

    def read_sql(self, query: str, connection: Any, **kwargs: Any) -> pd.DataFrame:
        return pd.read_sql(query, connection, **kwargs)

    # ── Processing ──────────────────────────────────────────────────

    def ensure_table(self, df: Any) -> pd.DataFrame:
        if isinstance(df, pd.DataFrame):
            return df
        return self.to_pandas(df)

    def to_pandas(self, df: Any, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        if isinstance(df, pd.DataFrame):
            return df[list(cols)].copy() if cols else df.copy()

        # Try polars
        try:
            import polars as pl
            if isinstance(df, pl.DataFrame):
                out = df.select(cols) if cols else df
                return out.to_pandas()
            if hasattr(df, "collect"):  # LazyFrame
                lf = df.select(cols) if cols else df
                return lf.collect().to_pandas()
        except Exception:
            pass

        # Try spark
        try:
            if hasattr(df, "toPandas"):
                sdf = df.select(*cols) if cols else df
                return sdf.toPandas()
        except Exception:
            pass

        raise TypeError("Unsupported table type for to_pandas")

    def schema_text(self, df: Any) -> str:
        if not isinstance(df, pd.DataFrame):
            try:
                df = self.to_pandas(df)
            except Exception:
                return "schema: <unknown>"
        dtypes = {c: str(t) for c, t in df.dtypes.items()}
        return (
            f"engine=pandas\n"
            f"rows={len(df)}, cols={len(df.columns)}\n"
            f"columns={list(df.columns)}\n"
            f"dtypes={dtypes}\n"
            f"missing={df.isna().sum().to_dict()}"
        )

    def exec_env(self, df: Any) -> Dict[str, Any]:
        env: Dict[str, Any] = {
            "df": df,
            "pd": pd,
            "engine": self.name,
            "to_pandas": self.to_pandas,
        }
        try:
            import matplotlib.pyplot as plt
            env["plt"] = plt
        except Exception:
            env["plt"] = None
        return env


# ═══════════════════════════════════════════════════════════════════════
#  Polars
# ═══════════════════════════════════════════════════════════════════════

class PolarsEngine(BaseEngine):
    name: EngineName = "polars"

    def _import_polars(self):
        import polars as pl
        return pl

    # ── Loading ─────────────────────────────────────────────────────

    def read_csv(self, path: str, **kwargs: Any) -> Any:
        pl = self._import_polars()
        return pl.scan_csv(path, **kwargs)

    def read_parquet(self, path: str, **kwargs: Any) -> Any:
        pl = self._import_polars()
        return pl.scan_parquet(path, **kwargs)

    def read_excel(self, path: str, **kwargs: Any) -> Any:
        pl = self._import_polars()
        return pl.read_excel(path, **kwargs).lazy()

    def read_sql(self, query: str, connection: Any, **kwargs: Any) -> Any:
        pl = self._import_polars()
        df = pd.read_sql(query, connection, **kwargs)
        return pl.from_pandas(df).lazy()

    # ── Processing ──────────────────────────────────────────────────

    def ensure_table(self, df: Any) -> Any:
        pl = self._import_polars()
        if isinstance(df, pl.DataFrame):
            return df.lazy()
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df).lazy()
        if hasattr(df, "collect") and hasattr(df, "schema"):
            return df  # already a LazyFrame
        raise TypeError("Cannot convert df to polars LazyFrame")

    def to_pandas(self, df: Any, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        pl = self._import_polars()
        if isinstance(df, pl.DataFrame):
            out = df.select(cols) if cols else df
            return out.to_pandas()
        if hasattr(df, "collect"):  # LazyFrame
            lf = df.select(cols) if cols else df
            return lf.collect().to_pandas()
        if isinstance(df, pd.DataFrame):
            return df[list(cols)].copy() if cols else df.copy()
        raise TypeError("Unsupported table type for to_pandas")

    def schema_text(self, df: Any) -> str:
        try:
            pl = self._import_polars()
            lf = df
            if isinstance(df, pl.DataFrame):
                lf = df.lazy()
            schema = getattr(lf, "schema", None)
            cols = list(schema.keys()) if schema else []
            dtypes = {k: str(v) for k, v in (schema or {}).items()}
            return (
                "engine=polars\n"
                f"cols={len(cols)}\n"
                f"columns={cols}\n"
                f"dtypes={dtypes}"
            )
        except Exception:
            return "engine=polars\nschema: <unavailable>"

    def exec_env(self, df: Any) -> Dict[str, Any]:
        pl = self._import_polars()
        env: Dict[str, Any] = {
            "df": df,
            "pd": pd,
            "pl": pl,
            "pl_df": df,
            "engine": self.name,
            "to_pandas": self.to_pandas,
        }
        try:
            import matplotlib.pyplot as plt
            env["plt"] = plt
        except Exception:
            env["plt"] = None
        return env


# ═══════════════════════════════════════════════════════════════════════
#  Spark
# ═══════════════════════════════════════════════════════════════════════

class SparkEngine(BaseEngine):
    name: EngineName = "spark"

    def _get_spark(self):
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()

    # ── Loading ─────────────────────────────────────────────────────

    def read_csv(self, path: str, **kwargs: Any) -> Any:
        spark = self._get_spark()
        header = kwargs.pop("header", True)
        infer = kwargs.pop("inferSchema", True)
        return spark.read.csv(path, header=header, inferSchema=infer, **kwargs)

    def read_parquet(self, path: str, **kwargs: Any) -> Any:
        spark = self._get_spark()
        return spark.read.parquet(path, **kwargs)

    def read_excel(self, path: str, **kwargs: Any) -> Any:
        # Spark has no built-in Excel reader — fall back via pandas
        pdf = pd.read_excel(path, **kwargs)
        spark = self._get_spark()
        return spark.createDataFrame(pdf)

    def read_sql(self, query: str, connection: Any, **kwargs: Any) -> Any:
        spark = self._get_spark()
        if isinstance(connection, str):
            # JDBC connection string
            return spark.read.format("jdbc").option("url", connection).option("query", query).load()
        # Fall back via pandas
        pdf = pd.read_sql(query, connection, **kwargs)
        return spark.createDataFrame(pdf)

    # ── Processing ──────────────────────────────────────────────────

    def ensure_table(self, df: Any) -> Any:
        if hasattr(df, "select") and hasattr(df, "schema") and hasattr(df, "limit"):
            return df  # already a Spark DataFrame
        if isinstance(df, pd.DataFrame):
            spark = self._get_spark()
            return spark.createDataFrame(df)
        raise TypeError("Cannot convert df to Spark DataFrame")

    def to_pandas(self, df: Any, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        if isinstance(df, pd.DataFrame):
            return df[list(cols)].copy() if cols else df.copy()
        if hasattr(df, "toPandas"):
            sdf = df.select(*cols) if cols else df
            return sdf.toPandas()
        raise TypeError("Unsupported table type for to_pandas")

    def schema_text(self, df: Any) -> str:
        try:
            fields = getattr(df.schema, "fields", [])
            cols = [f.name for f in fields]
            dtypes = {f.name: f.dataType.simpleString() for f in fields}
            return (
                "engine=spark\n"
                f"cols={len(cols)}\n"
                f"columns={cols}\n"
                f"dtypes={dtypes}"
            )
        except Exception:
            return "engine=spark\nschema: <unavailable>"

    def exec_env(self, df: Any) -> Dict[str, Any]:
        env: Dict[str, Any] = {
            "df": df,
            "pd": pd,
            "spark_df": df,
            "engine": self.name,
            "to_pandas": self.to_pandas,
        }
        try:
            from pyspark.sql import functions as F
            env["F"] = F
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt
            env["plt"] = plt
        except Exception:
            env["plt"] = None
        return env


# ═══════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════

_ENGINE_REGISTRY: Dict[str, type[BaseEngine]] = {
    "pandas": PandasEngine,
    "polars": PolarsEngine,
    "spark": SparkEngine,
}


def create_engine(name: EngineName) -> BaseEngine:
    """Factory method — instantiate the engine by name."""
    cls = _ENGINE_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown engine: {name!r}. Choose from {list(_ENGINE_REGISTRY)}")
    return cls()
