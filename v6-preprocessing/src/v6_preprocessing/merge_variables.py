import pandas as pd

from vantage6.algorithm.decorator.action import preprocessing
from vantage6.algorithm.tools.exceptions import DataError, UserInputError
from vantage6.algorithm.tools.util import info, error

from .utils import is_category

@preprocessing
def merge_variables(
    df: pd.DataFrame,
    column1: str,
    column2: str,
    output_column: str,
) -> pd.DataFrame:
    """
    Merge two categorical variables into one.
    """
    old_df = df.copy()

    if not is_category(df[column1]):
        error(f"Column {column1} is not a category. Returning original dataframe.")
        return df
    if not is_category(df[column2]):
        error(f"Column {column2} is not a category. Returning original dataframe.")
        return df
        
    info(f"Merging variables {column1} and {column2} into {output_column}.")

    try:
        merged = df[column1].astype(str) + "_" + df[column2].astype(str)
        df[output_column] = pd.Categorical(merged)
    except Exception as exc:
        print("FAILED TO PROCESS MERGE VARIABLES")
        print("Cant exit badly as it will render the dataframe unusable")
        print(exc)
        return old_df

    info(f"Done.")

    return df