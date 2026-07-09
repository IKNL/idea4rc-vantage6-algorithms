import math

import pandas as pd

from .type_guards import Idea4rcDType, classify_idea4rc_dtype


def to_datetime(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert columns to UTC datetime and strip time-of-day."""
    for column in columns:
        df[column] = pd.to_datetime(df[column], errors="coerce", utc=True).dt.normalize()
    return df


def to_category(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert columns to pandas categorical dtype."""
    for column in columns:
        df[column] = df[column].astype("category")
    return df


def _parse_float(value) -> float:
    """Parse any value to float; return NaN for missing/unparseable."""
    try:
        return float(str(value))
    except (ValueError, TypeError):
        return float("nan")


def to_int64(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert columns to nullable Int64 dtype."""
    for column in columns:
        df[column] = pd.array(
            [pd.NA if math.isnan(f := _parse_float(x)) else int(f) for x in df[column]],
            dtype="Int64",
        )
    return df


def to_float64(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert columns to nullable Float64 dtype."""
    for column in columns:
        df[column] = pd.array(
            [pd.NA if math.isnan(f := _parse_float(x)) else f for x in df[column]],
            dtype="Float64",
        )
    return df


def to_boolean(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert columns to nullable boolean dtype."""
    for column in columns:
        df[column] = df[column].astype("boolean")
    return df


def boolean_to_labeled_category(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert boolean columns to self-labeled categorical columns.

    Each value becomes '{col}=true', '{col}=false', or '{col}=N/A' (missing),
    then the column is cast to categorical. Non-boolean columns are left
    untouched, so the caller can safely pass a mixed list of columns.
    """
    for column in columns:
        if classify_idea4rc_dtype(df[column]) == Idea4rcDType.BOOLEAN:
            labeled = df[column].map({True: f"{column}=true", False: f"{column}=false"})
            df[column] = labeled.fillna(f"{column}=N/A").astype("category")
    return df
