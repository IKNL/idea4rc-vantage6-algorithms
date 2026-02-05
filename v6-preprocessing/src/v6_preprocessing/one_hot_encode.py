import pandas as pd

from vantage6.algorithm.decorator.action import preprocessing
from vantage6.algorithm.tools.exceptions import DataError, UserInputError

@preprocessing
def one_hot_encode(
    df: pd.DataFrame,
    column: str,
    categories: list[str],
    unknown_category: str | None = "unknown",
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

    # Map unseen categories to the unknown_category label
    df_copy = df.copy()
    df_copy[column] = df_copy[column].apply(
        lambda x: x if x in categories else unknown_category
    )

    # Perform one-hot encoding
    one_hot_df = pd.get_dummies(df_copy[column], prefix=prefix)

    # Merge one-hot encoded DataFrame with the original DataFrame
    df_out = pd.concat([df, one_hot_df], axis=1)

    # Drop the original column if specified
    if drop_original:
        df_out.drop(column, axis=1, inplace=True)

    return df_out