# Modified version of https://github.com/vantage6/vantage6/blob/release/5.0/vantage6-algorithm-tools/vantage6/algorithm/preprocessing/datetime.py

from vantage6.algorithm.decorator.action import preprocessing
import pandas as pd

from vantage6.algorithm.tools.exceptions import DataError, UserInputError


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
    try:
        dates = pd.to_datetime(df[column], format=fmt)
        dates_tz = dates.dt.tz

        if to_date_column:
            try:
                to_date = pd.to_datetime(df[to_date_column], format=fmt)
            except ValueError as exc:
                raise DataError(
                    f"The column `{to_date_column}` cannot be converted to a datetime "
                    "object."
                ) from exc
            # Normalize timezone awareness to match dates
            if dates_tz is None and to_date.dt.tz is not None:
                # Convert aware to_date to naive by converting to UTC then removing timezone
                # Convert to UTC first, then create naive timestamps from string representation
                to_date = pd.to_datetime(to_date.dt.tz_convert("UTC").astype(str))
            elif dates_tz is not None and to_date.dt.tz is None:
                # Convert naive to_date to aware in same timezone as dates
                to_date = to_date.dt.tz_localize("UTC").dt.tz_convert(dates_tz)
            elif dates_tz is not None and to_date.dt.tz is not None and to_date.dt.tz != dates_tz:
                # Both aware but different timezones, convert to_date to match dates
                to_date = to_date.dt.tz_convert(dates_tz)
            duration_col = (to_date - dates).dt.days
        elif to_date:
            try:
                to_date = pd.Timestamp(to_date)
            except ValueError as exc:
                raise UserInputError("The `to_date` must be a valid date string.") from exc
            # Normalize timezone awareness to match dates
            if dates_tz is None:
                # Make to_date naive if it's aware
                if to_date.tz is not None:
                    # Convert to UTC first, then create naive timestamp
                    to_date_utc = to_date.tz_convert("UTC")
                    to_date = pd.Timestamp(to_date_utc.to_pydatetime().replace(tzinfo=None))
            else:
                # Make to_date aware in the same timezone as dates
                if to_date.tz is None:
                    to_date = to_date.tz_localize("UTC").tz_convert(dates_tz)
                elif to_date.tz != dates_tz:
                    to_date = to_date.tz_convert(dates_tz)
            duration_col = (to_date - dates).dt.days
        else:
            to_date = pd.to_datetime("today")
            # Normalize timezone awareness to match dates
            if dates_tz is not None:
                # Make to_date aware in the same timezone as dates
                to_date = pd.Timestamp(to_date).tz_localize("UTC").tz_convert(dates_tz)
            duration_col = (to_date - dates).dt.days

        df[output_column] = duration_col
    except Exception as exc:
        print("FAILED TO PROCESS TIME DELTA")
        print("Cant exit badly as it will render the dataframe unusable")
        print(exc)
        return old_df

    return df