from datetime import timedelta

import pandas as pd

from vantage6.algorithm.decorator.action import preprocessing
from vantage6.algorithm.tools.util import info, error

SURGERY = [
    "date",
    "intent",
    "margins_after_surgery",
    # head and neck
    "extra_nodal_extension",
    "laterality_of_the_dissection",
    "neck_surgery",
    "date_of_neck_surgery",
    "surgery_hospital",
]

SYSTEMIC_TREATMENT = [
    "start_date",
    "end_date",
    "regimen",
    "setting",
    "type",
    "for_end_of_treatment",
]

RADIO = [
    "start_date",
    "end_date",
    "hospital",
    "setting",
    "intent",
    "total_dose_gy",
    "number_of_fractions",
    "treatment_completed_as_planned",
    "intraoperative_radio",
]

mapper_ = {"radio": RADIO, "surgery": SURGERY, "systemic_treatment": SYSTEMIC_TREATMENT}
DATE_KEY = {"surgery": "date", "radio": "start_date", "systemic_treatment": "start_date"}
MAX_EVENTS = 5


def _resolve_date(date_string, date_column, row):
    if date_string is not None and date_column is not None:
        raise ValueError("Cannot supply both date_string and date_column")
    if date_string is None and date_column is None:
        raise ValueError("Must supply either date_string or date_column")

    if date_string is not None:
        return pd.Timestamp(date_string, tz="UTC")
    else:
        return row[date_column]


def _find_event_index_by_date(row, type_, date_var, start_ts, end_ts):
    for i in range(1, MAX_EVENTS + 1):
        col_name = f"{type_}_{i}_{date_var}"
        if col_name not in row.index:
            continue
        event_date = row[col_name]
        if pd.isna(event_date):
            continue
        if start_ts is not None and event_date < start_ts:
            continue
        if end_ts is not None and event_date > end_ts:
            continue
        return i
    return None


def _find_last_event_index(row, type_):
    for i in range(MAX_EVENTS, 0, -1):
        col_name = f"{type_}_{i}_{DATE_KEY[type_]}"
        if col_name not in row.index:
            continue
        if pd.notna(row[col_name]):
            return i
    return None


def _copy_event_columns(df, type_, event_index_series, dest_name):
    vars = mapper_[type_]
    for var in vars:
        df[f"{dest_name}_{var}"] = pd.NA

    for idx in df.index:
        event_idx = event_index_series[idx]
        if event_idx is None:
            continue
        for var in vars:
            src_col = f"{type_}_{event_idx}_{var}"
            dest_col = f"{dest_name}_{var}"
            if src_col in df.columns:
                df.loc[idx, dest_col] = df.loc[idx, src_col]


@preprocessing
def annotate_event_by_index(
    df: pd.DataFrame, type_: str, index: int | str, name: str
) -> pd.DataFrame:

    old_df = df.copy()

    try:
        if index == "first":
            index = 1

        if isinstance(index, str) and index == "last":
            event_indices = df.apply(lambda row: _find_last_event_index(row, type_), axis=1)
            _copy_event_columns(df, type_, event_indices, f"{name}_{type_}")
        else:
            vars = mapper_[type_]
            for var in vars:
                src = f"{type_}_{index}_{var}"
                dest = f"{name}_{type_}_{var}"
                if src in df.columns:
                    df[dest] = df[src]

        info(f"Annotated {type_} index {index} as {name}_{type_}")
        return df

    except Exception as exc:
        error(f"Failed to annotate event by index: {exc}")
        error(exc)
        return old_df


@preprocessing
def annotate_event_by_date_range(
    df: pd.DataFrame,
    type_: str,
    name: str,
    start_date_string: str | None = None,
    start_date_column: str | None = None,
    end_date_string: str | None = None,
    end_date_column: str | None = None,
    date_var: str | None = None,
) -> pd.DataFrame:

    old_df = df.copy()

    try:
        if date_var is None:
            date_var = DATE_KEY[type_]

        if start_date_string is not None and start_date_column is not None:
            error("Cannot supply both start_date_string and start_date_column")
            return old_df

        if end_date_string is not None and end_date_column is not None:
            error("Cannot supply both end_date_string and end_date_column")
            return old_df

        def get_event_index(row):
            start_ts = None
            end_ts = None

            if start_date_string is not None:
                start_ts = pd.Timestamp(start_date_string, tz="UTC")
            elif start_date_column is not None:
                start_ts = row[start_date_column]

            if end_date_string is not None:
                end_ts = pd.Timestamp(end_date_string, tz="UTC")
            elif end_date_column is not None:
                end_ts = row[end_date_column]

            return _find_event_index_by_date(row, type_, date_var, start_ts, end_ts)

        event_indices = df.apply(get_event_index, axis=1)
        _copy_event_columns(df, type_, event_indices, f"{name}_{type_}")

        num_found = (event_indices is not None).sum()
        info(f"Found {num_found} events in date range")

        if num_found == 0:
            error(f"No events found for {type_} in date range")

        return df

    except Exception as exc:
        error(f"Failed to annotate event by date range: {exc}")
        error(exc)
        return old_df


@preprocessing
def annotate_event_within_window(
    df: pd.DataFrame,
    type_: str,
    name: str,
    reference_date_string: str | None = None,
    reference_date_column: str | None = None,
    days_before: int = 0,
    days_after: int = 0,
    date_var: str | None = None,
) -> pd.DataFrame:

    old_df = df.copy()

    try:
        if reference_date_string is not None and reference_date_column is not None:
            error("Cannot supply both reference_date_string and reference_date_column")
            return old_df

        if reference_date_string is None and reference_date_column is None:
            error("Must supply either reference_date_string or reference_date_column")
            return old_df

        if date_var is None:
            date_var = DATE_KEY[type_]

        def get_event_index(row):
            ref_ts = _resolve_date(reference_date_string, reference_date_column, row)
            if pd.isna(ref_ts):
                return None

            start_ts = ref_ts - timedelta(days=days_before)
            end_ts = ref_ts + timedelta(days=days_after)

            return _find_event_index_by_date(row, type_, date_var, start_ts, end_ts)

        event_indices = df.apply(get_event_index, axis=1)
        _copy_event_columns(df, type_, event_indices, f"{name}_{type_}")

        num_found = (event_indices is not None).sum()
        info(f"Found {num_found} events within window")

        return df

    except Exception as exc:
        error(f"Failed to annotate event within window: {exc}")
        error(exc)
        return old_df
