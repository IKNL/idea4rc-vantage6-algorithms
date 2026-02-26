import pandas as pd

def is_boolean(column: pd.Series) -> bool:
    """
    This function checks if the given value is a boolean.

    Parameters
    ----------
    column : pd.Series
        The column to check if it is a boolean.

    Returns
    -------
    bool
        True if the column is a boolean, False otherwise.
    """
    return pd.api.types.is_bool_dtype(column)

def is_category(column: pd.Series) -> bool:
    """
    This function checks if the given column is a category.

    Parameters
    ----------
    column : pd.Series
        The column to check if it is a category.

    Returns
    -------
    bool
        True if the column is a category, False otherwise.
    """
    return pd.api.types.is_categorical_dtype(column)

def is_datetime(column: pd.Series) -> bool:
    """
    This function checks if the given column is a datetime.

    Parameters
    ----------
    column : pd.Series
        The column to check if it is a datetime.

    Returns
    -------
    bool
        True if the column is a datetime, False otherwise.
    """
    return pd.api.types.is_datetime64_any_dtype(column)

def is_float(column: pd.Series) -> bool:
    """
    This function checks if the given column is a float64.

    Parameters
    ----------
    column : pd.Series
        The column to check if it is a float64.

    Returns
    -------
    bool
        True if the column is a float64, False otherwise.
    """
    return pd.api.types.is_float_dtype(column)

def is_int(column: pd.Series) -> bool:
    """
    This function checks if the given column is a int64.
    """
    return pd.api.types.is_int64_dtype(column)