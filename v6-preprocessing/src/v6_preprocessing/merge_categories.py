import pandas as pd

from vantage6.algorithm.decorator.action import preprocessing
from vantage6.algorithm.tools.exceptions import DataError, UserInputError

@preprocessing
def merge_categories(
    df: pd.DataFrame,
    column: str,
    output_column: str,
    mapping: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Merge categories of a column into a single category.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        The column name to merge categories of.
    output_column : str
        The new column name to store the merged categories.
    mapping : dict[str, list[str]]
        A dictionary mapping the categories to merge to the new category.

    Returns
    -------
    pd.DataFrame
        DataFrame with the merged categories.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"category": ["a", "b", "c", "d"]})
    >>> merge_categories(df, "category", "merged_category", {"a": ["b", "c"], "d": ["e", "f", "g"]})
    >>> df["merged_category"]
    >>> df[["merged_category"]]
    category  merged_category
    0        a             b
    1        b             b
    2        c             b
    3        d             g
    """

    old_df = df.copy()
    try:
        df[output_column] = df[column].replace({v: k for k, vals in mapping.items() for v in vals})
    except Exception as exc:
        print("FAILED TO PROCESS MERGE CATEGORIES")
        print("Cant exit badly as it will render the dataframe unusable")
        print(exc)
        return old_df

    return df
