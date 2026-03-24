import pandas as pd


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


def to_int64(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert columns to nullable Int64 dtype."""
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    return df


def to_float64(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert columns to nullable Float64 dtype."""
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Float64")
    return df


def to_boolean(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert columns to nullable boolean dtype."""
    for column in columns:
        df[column] = df[column].astype("boolean")
    return df
