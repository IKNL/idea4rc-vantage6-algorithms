import pandas as pd

from vantage6.algorithm.decorator.action import preprocessing
from vantage6.algorithm.tools.exceptions import DataError, UserInputError
from vantage6.algorithm.tools.util import info, error



from .utils import is_category

@preprocessing
def to_boolean(
    df: pd.DataFrame,
    column: str,
    output_column: str,
    true_values: list[str] | None = None,
    false_values: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert a categorical variable to a boolean variable.
    """

    if not is_category(df[column]):
        error(f"Column {column} is not a category. Returning original dataframe.")
        return df

    if (true_values is None and false_values is None):
        error("True values and false values are not provided. Returning original dataframe.")
        return df
    
    if (true_values and false_values):
        error("True values and false values are provided. Returning original dataframe.")
        return df

    info(f"Converting column {column} to boolean.")
    info(f"True values: {true_values}")
    info(f"False values: {false_values}")
    info(f"Output column: {output_column}")

    try:
        if true_values:
            df[output_column] = (df[column].isin(true_values)).astype("boolean")
        elif false_values:
            df[output_column] = (~df[column].isin(false_values)).astype("boolean")
    except Exception as exc:
        error(f"Failed to convert column {column} to boolean. Returning original dataframe.")
        error(exc)
        return df

    info("Done.")
    return df