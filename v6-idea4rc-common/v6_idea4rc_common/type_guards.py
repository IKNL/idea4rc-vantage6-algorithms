from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import pandas as pd


class Idea4rcDType(str, Enum):
    INT64 = "Int64"
    FLOAT64 = "Float64"
    BOOLEAN = "boolean"
    CATEGORY = "category"
    DATETIME64TZ = "datetime64[ns, tz]"


@dataclass
class ColumnTypeError(ValueError):
    algorithm: str
    column: str
    expected: str
    actual: str

    def __str__(self) -> str:  # pragma: no cover (presentation only)
        return (
            f"[{self.algorithm}] Column '{self.column}' has dtype '{self.actual}', "
            f"expected {self.expected}."
        )


def _dtype_str(series: pd.Series) -> str:
    try:
        return str(series.dtype)
    except Exception:
        return "<unknown>"


def classify_idea4rc_dtype(series: pd.Series) -> Idea4rcDType | None:
    """
    Classify a series into one of the IDEA4RC accepted dtypes.

    Strict mapping (only exact accepted types):
    - Int64 (nullable)  -> Idea4rcDType.INT64
    - Float64 (nullable) -> Idea4rcDType.FLOAT64
    - boolean (nullable) -> Idea4rcDType.BOOLEAN
    - category -> Idea4rcDType.CATEGORY
    - datetime64[ns, tz] (tz-aware) -> Idea4rcDType.DATETIME64TZ
    """
    dtype = series.dtype

    # Use string form for strictness on pandas extension dtypes
    dtype_name = str(dtype)

    if dtype_name == Idea4rcDType.INT64.value:
        return Idea4rcDType.INT64
    if dtype_name == Idea4rcDType.FLOAT64.value:
        return Idea4rcDType.FLOAT64
    if dtype_name == Idea4rcDType.BOOLEAN.value:
        return Idea4rcDType.BOOLEAN

    if pd.api.types.is_categorical_dtype(dtype):
        return Idea4rcDType.CATEGORY

    # tz-aware datetime only
    if pd.api.types.is_datetime64tz_dtype(dtype):
        return Idea4rcDType.DATETIME64TZ

    return None


def assert_columns_exist(df: pd.DataFrame, columns: Sequence[str], *, algorithm: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ColumnTypeError(
            algorithm=algorithm,
            column=",".join(missing),
            expected="present in dataframe",
            actual="missing",
        )


def assert_column_dtype_in(
    df: pd.DataFrame,
    column: str,
    allowed: Iterable[Idea4rcDType],
    *,
    algorithm: str,
    expected_kind: str | None = None,
) -> None:
    if column not in df.columns:
        raise ColumnTypeError(
            algorithm=algorithm,
            column=column,
            expected="present in dataframe",
            actual="missing",
        )
    actual = classify_idea4rc_dtype(df[column])
    allowed_set = set(allowed)
    if actual not in allowed_set:
        expected = (
            expected_kind
            if expected_kind is not None
            else f"one of: {', '.join(sorted([a.value for a in allowed_set]))}"
        )
        raise ColumnTypeError(
            algorithm=algorithm,
            column=column,
            expected=expected,
            actual=_dtype_str(df[column]),
        )


def assert_columns_dtype_in(
    df: pd.DataFrame,
    columns: Sequence[str],
    allowed: Iterable[Idea4rcDType],
    *,
    algorithm: str,
    expected_kind: str | None = None,
) -> None:
    for col in columns:
        assert_column_dtype_in(
            df, col, allowed, algorithm=algorithm, expected_kind=expected_kind
        )


def is_binary_int64_01(series: pd.Series) -> bool:
    """
    True iff dtype is exactly Int64 and non-null values are subset of {0, 1}.
    """
    if str(series.dtype) != Idea4rcDType.INT64.value:
        return False
    values = series.dropna().unique()
    return set(values).issubset({0, 1})


def convert_int64_01_to_boolean(series: pd.Series, *, algorithm: str, column: str) -> pd.Series:
    """
    Convert Int64 {0,1,NA} to pandas nullable boolean.
    """
    if not is_binary_int64_01(series):
        raise ColumnTypeError(
            algorithm=algorithm,
            column=column,
            expected="Int64 with values restricted to {0, 1, NA}",
            actual=_dtype_str(series),
        )
    return series.astype("boolean")

