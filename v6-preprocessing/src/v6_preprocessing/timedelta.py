# Modified version of https://github.com/vantage6/vantage6/blob/release/5.0/vantage6-algorithm-tools/vantage6/algorithm/preprocessing/datetime.py

from vantage6.algorithm.decorator.action import preprocessing
import pandas as pd

from vantage6.algorithm.tools.util import info, error
from .utils import is_datetime

@preprocessing
def timedelta(
    df: pd.DataFrame,
    column: str,
    output_column: str = "timedelta",
    to_date_column: str | None = None,
    to_date: str | None = None,
    fmt: str | None = None,
) -> pd.DataFrame:
    """
    Create a timedelta column from a datetime column. The new column shows the time in
    days since the reference date.

    A reference column may be provided to calculate the timedelta for each row.
    Otherwise, the default reference date is today.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        The name of the datetime column to convert to a timedelta.
    output_column : str
        Output column name.
    to_date_column : str | None, optional
        A column containing dates to which the timedelta is calculated for each
        row. If not provided, `to_date` is used for all rows.
    to_date : str | None, optional
        The date to which the timedelta is calculated. Defaults to today if not
        provided. Ignored if `to_date_column` is provided.
    fmt : str | None, optional
        The format to use for parsing date strings if the `column` or
        `to_date_column` contains strings instead of actual datetime objects.
        If None, pandas will infer the format.

    Returns
    -------
    pd.DataFrame
        DataFrame with the timedelta column in days.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"date": [pd.Timestamp("2021-01-01"),
    ... pd.Timestamp("2021-02-01")], "ref": [pd.Timestamp("2021-01-15"),
    ... pd.Timestamp("2021-02-15")]})
    >>> timedelta(df, "date", "days_to_ref", to_date_column="ref")
            date        ref  days_to_ref
    0 2021-01-01 2021-01-15           14
    1 2021-02-01 2021-02-15           14

    >>> today = pd.to_datetime("today")
    >>> df = pd.DataFrame({"birthdate": [today - pd.Timedelta(days=300),
    ... today - pd.Timedelta(days=250)]})
    >>> df['birthdate'] = df['birthdate'].dt.date
    >>> df = timedelta(df, "birthdate", "age_in_days")
    >>> df[['age_in_days']]
       age_in_days
    0          300
    1          250

    """
    old_df = df.copy()

    if not is_datetime(df[column]):
        error(f"Column {column} is not a datetime. Returning original dataframe.")
        return old_df

    if to_date_column and not is_datetime(df[to_date_column]):
        error(f"Column {to_date_column} is not a datetime. Returning original dataframe.")
        return old_df

    try:

        info(f"Calculating timedelta for column {column} into {output_column}.")

        dates = df[column]

        if to_date_column:
            info(f"Using column {to_date_column} as reference date.")
            to_date = df[to_date_column]
        elif to_date:
            info(f"Using date {to_date} as reference date.")
            try:
                to_date = pd.Timestamp(to_date, tz="UTC")
            except ValueError as exc:
                error("The `to_date` must be a valid date string.")
        else:
            info(f"Using today as reference date.")
            to_date = pd.to_datetime("today", utc=True)

        info(f"Calculating timedelta...")
        df[output_column] = (to_date - dates).dt.days

    except Exception as exc:
        error("FAILED TO PROCESS TIME DELTA")
        error("Cant exit badly as it will render the dataframe unusable")
        error(exc)
        return old_df

    return df