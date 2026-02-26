import pandas as pd

from vantage6.algorithm.decorator.action import preprocessing
from vantage6.algorithm.tools.util import info, error

@preprocessing
def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Drop a column from a dataframe.
    """
    old_df = df.copy()
    if column not in df.columns:
        error(f"Column {column} not found in dataframe. Returning original dataframe.")
        return old_df
    
    try:
        info(f"Dropping column {column} from dataframe.")
        df.drop(columns=[column], inplace=True)
    except Exception as exc:
        error(f"Failed to drop column {column} from dataframe. Returning original dataframe.")
        error(exc)
        return old_df
    
    info(f"Done.")
    return df