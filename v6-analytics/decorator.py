from functools import wraps

import pandas as pd

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import _get_user_database_labels


def new_data_decorator(func: callable, *args, **kwargs) -> callable:
    """
    Decorator to add data to the function.

    This returns the function with the `data_frames` and `cohort_names` as the
    first two arguments.
    """

    @wraps(func)
    def decorator(*args, mock_data: list[pd.DataFrame] = None, **kwargs) -> callable:

        if mock_data:
            data_frames = mock_data
            cohort_names = [f"cohort_{i}" for i in range(len(mock_data))]
        else:
            cohort_names = _get_user_database_labels()
            data_frames = []
            for cohort_name in cohort_names:
                info(f"Loading data for cohort {cohort_name}")

                df = pd.read_parquet(f"/mnt/data/{cohort_name}.parquet")
                data_frames.append(df)

        args = (data_frames, cohort_names, *args)
        return func(*args, **kwargs)

    decorator.wrapped_in_data_decorator = True
    return decorator
