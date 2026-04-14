import pandas as pd

from vantage6.algorithm.decorator.action import preprocessing
from vantage6.algorithm.tools.util import info, error

from .utils import is_category


@preprocessing
def one_hot_encode(
    df: pd.DataFrame,
    column: str,
    # categories: list[str],
    # unknown_category: str | None = "unknown",
    drop_original: bool = False,
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Perform one-hot encoding on a specific column of a DataFrame.

    As one node may not have access to all possible categories in the entire dataset,
    this requires predefined categories to be specified upfront. The function allows
    encoding of unseen categories into a specified 'unknown' category label.
    The original column can be optionally dropped, and a prefix can be added
    to the new one-hot encoded columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to encode.
    column : str
        The column to one-hot encode.
    categories : list of str
        List of predefined categories.
    unknown_category : str | None, optional
        Label for unseen categories.
    drop_original : bool, optional
        Whether to drop the original column, default is False.
    prefix : str | None, optional
        Prefix for the new one-hot encoded columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one-hot encoded column.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'color': ['red', 'green', 'blue', 'yellow']
    ... })

    >>> one_hot_encode(df, 'color', ['red', 'green', 'blue'])
       blue  green  red  unknown
    0     0      0    1        0
    1     0      1    0        0
    2     1      0    0        0
    3     0      0    0        1

    >>> one_hot_encode(df, 'color', ['red', 'green'], drop_original=False)
        color  green  red  unknown
    0     red      0    1        0
    1   green      1    0        0
    2    blue      0    0        1
    3  yellow      0    0        1

    >>> one_hot_encode(df, 'color', ['red', 'green'], unknown_category='other',
    ...                prefix='col')
       col_green  col_other  col_red
    0          0          0        1
    1          1          0        0
    2          0          1        0
    3          0          1        0

    """

    old_df = df.copy()
    
    if not is_category(df[column]):
        error(f"Column {column} is not a category. Returning original dataframe.")
        return old_df

    info(f"One-hot encoding column {column}.")
    info(f"Prefix: {prefix}")
    info(f"Drop original: {drop_original}")

    try:
        # Map unseen categories to the unknown_category label
        df_copy = df.copy()
        # df_copy[column] = df_copy[column].apply(
        #     lambda x: x if x in categories else unknown_category
        # )

        # Perform one-hot encoding
        one_hot_df = pd.get_dummies(df_copy[column], prefix=prefix).astype("Int64")

        # Drop columns from df that will be replaced by one-hot columns (avoids duplicates, one_hot overwrites)
        cols_replaced = [c for c in one_hot_df.columns if c in df.columns]
        df_clean = df.drop(columns=cols_replaced)

        # Merge one-hot encoded DataFrame with the original DataFrame
        df_out = pd.concat([df_clean, one_hot_df], axis=1)

        # Drop the original column if specified
        if drop_original:
            df_out.drop(column, axis=1, inplace=True)
    except Exception as exc:
        print("FAILED TO PROCESS ONE HOT ENCODING")
        print("Cant exit badly as it will render the dataframe unusable")
        print(exc)
        return old_df

    info(f"Done.")

    return df_out