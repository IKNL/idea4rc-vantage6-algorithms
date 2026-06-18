import re

import pandas as pd


def _ensure_utc(s: pd.Series) -> pd.Series:
    """Localize tz-naive datetime Series to UTC; pass through tz-aware as-is.

    NaT-only columns created without timezone info (e.g. pd.NaT in a dict)
    are tz-naive. Localizing them keeps NaT as NaT and allows comparison with
    tz-aware (UTC) columns that all real IDEA4RC date columns use.
    """
    if not pd.api.types.is_datetime64_any_dtype(s):
        return s
    if getattr(s.dtype, "tz", None) is None:
        return s.dt.tz_localize("UTC")
    return s


# ---------------------------------------------------------------------------
# Column discovery — scans df.columns dynamically, no hardcoded slot counts
# ---------------------------------------------------------------------------

def _start_cols(df: pd.DataFrame, treatment: str) -> list[str]:
    if treatment == "surgery":
        return _date_cols(df, treatment)
    pattern = re.compile(rf"^{treatment}_\d+_start_date$")
    return sorted(c for c in df.columns if pattern.match(c))


def _end_cols(df: pd.DataFrame, treatment: str) -> list[str]:
    pattern = re.compile(rf"^{treatment}_\d+_end_date$")
    return sorted(c for c in df.columns if pattern.match(c))


def _date_cols(df: pd.DataFrame, treatment: str) -> list[str]:
    pattern = re.compile(rf"^{treatment}_\d+_date$")
    return sorted(c for c in df.columns if pattern.match(c))


def _slot_start(df: pd.DataFrame, treatment: str, n: int) -> str:
    """Return the column name for the n-th start date (1-based)."""
    if treatment == "surgery":
        return f"{treatment}_{n}_date"
    return f"{treatment}_{n}_start_date"


def _slot_end(df: pd.DataFrame, treatment: str, n: int) -> str:
    return f"{treatment}_{n}_end_date"


# ---------------------------------------------------------------------------
# Date utilities — all vectorized over pd.Series
# ---------------------------------------------------------------------------

def get_first_date(df: pd.DataFrame, col_names: list[str]) -> pd.Series:
    if not col_names:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    utc_cols = [_ensure_utc(df[c]) for c in col_names]
    return pd.concat(utc_cols, axis=1).min(axis=1)


def get_last_date(df: pd.DataFrame, col_names: list[str]) -> pd.Series:
    if not col_names:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    utc_cols = [_ensure_utc(df[c]) for c in col_names]
    return pd.concat(utc_cols, axis=1).max(axis=1)


def within_days_after(date: pd.Series, ref: pd.Series, days: int) -> pd.Series:
    """True when date <= ref + days (NaT on either side → False)."""
    return date.notna() & ref.notna() & (date <= ref + pd.Timedelta(days=days))


def within_days_before(date: pd.Series, ref: pd.Series, days: int) -> pd.Series:
    """True when date >= ref - days (NaT on either side → False)."""
    return date.notna() & ref.notna() & (date >= ref - pd.Timedelta(days=days))


def within_symmetric_gap(a: pd.Series, b: pd.Series, days: int) -> pd.Series:
    """True when |a - b| <= days (NaT on either side → False)."""
    return a.notna() & b.notna() & ((a - b).abs() <= pd.Timedelta(days=days))


def dates_overlap(s1: pd.Series, e1: pd.Series, s2: pd.Series, e2: pd.Series) -> pd.Series:
    """True when [s1, e1) overlaps [s2, e2): s1 < e2 AND e1 > s2."""
    return s1.notna() & e1.notna() & s2.notna() & e2.notna() & (s1 < e2) & (e1 > s2)


# ---------------------------------------------------------------------------
# Treatment presence / counting
# ---------------------------------------------------------------------------

def has_treatment(df: pd.DataFrame, treatment: str) -> pd.Series:
    cols = _start_cols(df, treatment)
    if not cols:
        return pd.Series(False, index=df.index, dtype="boolean")
    return df[cols].notna().any(axis=1).astype("boolean")


def count_treatments(df: pd.DataFrame, treatment: str) -> pd.Series:
    cols = _start_cols(df, treatment)
    if not cols:
        return pd.Series(0, index=df.index, dtype="Int64")
    return df[cols].notna().sum(axis=1).astype("Int64")


def first_start(df: pd.DataFrame, treatment: str) -> pd.Series:
    cols = _start_cols(df, treatment)
    return get_first_date(df, cols)


def last_end(df: pd.DataFrame, treatment: str) -> pd.Series:
    cols = _end_cols(df, treatment)
    if not cols and treatment == "surgery":
        return first_start(df, treatment)
    return get_last_date(df, cols)


# ---------------------------------------------------------------------------
# Slot-specific helpers
# ---------------------------------------------------------------------------

def nth_start(df: pd.DataFrame, treatment: str, n: int) -> pd.Series:
    col = _slot_start(df, treatment, n)
    if col not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    return _ensure_utc(df[col])


def nth_end(df: pd.DataFrame, treatment: str, n: int) -> pd.Series:
    col = _slot_end(df, treatment, n)
    if col not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    return _ensure_utc(df[col])


def chemo_n_overlaps_radio_m(df: pd.DataFrame, n: int, m: int) -> pd.Series:
    return dates_overlap(
        nth_start(df, "chemo", n),
        nth_end(df, "chemo", n),
        nth_start(df, "radio", m),
        nth_end(df, "radio", m),
    )


# ---------------------------------------------------------------------------
# Systemic treatment helpers (chemo OR immuno OR targeted)
# ---------------------------------------------------------------------------

def has_any_systemic(df: pd.DataFrame) -> pd.Series:
    return (
        has_treatment(df, "chemo")
        | has_treatment(df, "immuno")
        | has_treatment(df, "targeted")
    )


def count_systemic(df: pd.DataFrame) -> pd.Series:
    return (
        count_treatments(df, "chemo")
        + count_treatments(df, "immuno")
        + count_treatments(df, "targeted")
    )


def first_systemic_start(df: pd.DataFrame) -> pd.Series:
    candidates = [first_start(df, t).rename(t) for t in ("chemo", "immuno", "targeted")]
    return pd.concat(candidates, axis=1).min(axis=1)


def last_systemic_end(df: pd.DataFrame) -> pd.Series:
    candidates = [last_end(df, t).rename(t) for t in ("chemo", "immuno", "targeted")]
    return pd.concat(candidates, axis=1).max(axis=1)
