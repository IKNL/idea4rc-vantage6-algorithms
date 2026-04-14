import pandas as pd

from v6_idea4rc_common.type_guards import Idea4rcDType, classify_idea4rc_dtype

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
    return classify_idea4rc_dtype(column) == Idea4rcDType.BOOLEAN

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
    return classify_idea4rc_dtype(column) == Idea4rcDType.CATEGORY

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
    # Strict IDEA4RC accepted datetime dtype (tz-aware)
    return classify_idea4rc_dtype(column) == Idea4rcDType.DATETIME64TZ

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
    return classify_idea4rc_dtype(column) == Idea4rcDType.FLOAT64

def is_int(column: pd.Series) -> bool:
    """
    This function checks if the given column is a int64.
    """
    return classify_idea4rc_dtype(column) == Idea4rcDType.INT64