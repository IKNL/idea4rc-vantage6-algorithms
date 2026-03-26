import pandas as pd

from vantage6.algorithm.decorator.action import preprocessing
from vantage6.algorithm.tools.util import info, error

from .utils import is_int, is_float

@preprocessing
def basic_arithmetic(df: pd.DataFrame, column1: str | int | float, column2: str | int | float, operation: str, output_column: str) -> pd.DataFrame:
    """
    Add two columns together.
    """
    old_df = df.copy()
    
    if operation not in ["add", "subtract", "multiply", "divide"]:
        error(f"Operation {operation} is not supported. Returning original dataframe.")
        return old_df
    
    if isinstance(column1, str):
        column1 = df[column1]
        if not (is_int(column1) or is_float(column1)):
            error(f"Column 1 is not numeric. Returning original dataframe.")
            return old_df
    if isinstance(column2, str):
        column2_name = column2
        column2 = df[column2]
        if not (is_int(column2) or is_float(column2)):
            error(f"Column {column2_name} is not numeric. Returning original dataframe.")
            return old_df
    
    try:
        if operation == "add":
            df[output_column] = column1 + column2
        elif operation == "subtract":
            df[output_column] = column1 - column2
        elif operation == "multiply":
            df[output_column] = column1 * column2
        elif operation == "divide":
            df[output_column] = column1 / column2

        # Set the dtype to a nullable IDEA4RC pandas type
        if is_float(df[output_column]):
            df[output_column] = df[output_column].astype("Float64")
        else:
            df[output_column] = df[output_column].astype("Int64")

    except Exception as exc:
        error(f"Failed to add columns {column1} and {column2} into {output_column}. Returning original dataframe.")
        error(exc)
        return old_df
    
    info("Done.")
    return df