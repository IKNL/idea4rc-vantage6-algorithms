import pandas as pd

from vantage6.algorithm.decorator.action import preprocessing
from vantage6.algorithm.tools.exceptions import DataError, UserInputError

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
    try:
        merged = df[column1].astype(str) + "_" + df[column2].astype(str)
        df[output_column] = pd.Categorical(merged)
    except Exception as exc:
        print("FAILED TO PROCESS MERGE VARIABLES")
        print("Cant exit badly as it will render the dataframe unusable")
        print(exc)
        return old_df

    return df