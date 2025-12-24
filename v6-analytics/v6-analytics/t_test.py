import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from scipy.stats import t
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.decorator import (
    algorithm_client,
    central,
    dataframes,
    federated,
)
from vantage6.algorithm.tools.exceptions import InputError
from vantage6.algorithm.tools.util import get_env_var, info

T_TEST_MINIMUM_NUMBER_OF_RECORDS = 3

@central
@algorithm_client
def t_test_central(
    client: AlgorithmClient, 
    organizations_to_include: list[int]
) -> dict:
    """
    Send task to each node participating in the task to compute a local mean and sample
    variance, aggregate them to compute the t value for the independent sample t-test,
    and return the result.

    Parameters
    ----------
    client : AlgorithmClient
        The client object used to communicate with the server.
    organizations_to_include : list[int]
        The organizations to include in the task.

    Returns
    -------
    dict
        The `t` value for the independent-samples t test.
    """
    # create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        method="t_test_partial",
        arguments={},
        organizations=organizations_to_include,
        name="Subtask mean and sample variance",
        description="Compute mean and sample variance per data station.",
    )

    # wait for node to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")

    final_result = {}
    cohort_names = results[0].keys()
    for cohort_name in cohort_names:

        final_result[cohort_name] = {}

        columns_0 = results[0][cohort_name].keys()
        columns_1 = results[1][cohort_name].keys()
        columns = list(set(columns_0) & set(columns_1))
        for col in columns:
            count_0 = results[0][cohort_name][col]["count"]
            count_1 = results[1][cohort_name][col]["count"]
            
            # Check for insufficient data to perform t-test
            if count_0 == 0 or count_1 == 0:
                info(f"Skipping {col} in {cohort_name}: one or both cohorts have zero count")
                continue
            
            # Check if we have enough data for pooled variance calculation
            # Need at least 2 observations total (1 in each group) for dof > 0
            total_count = count_0 + count_1
            if total_count < 3:
                info(f"Skipping {col} in {cohort_name}: insufficient total count ({total_count})")
                continue
            
            # Compute pooled variance
            dof_denominator = total_count - 2
            if dof_denominator == 0:
                info(f"Skipping {col} in {cohort_name}: degrees of freedom denominator is zero")
                continue
            
            Sp = (
                (count_0 - 1) * results[0][cohort_name][col]["variance"]
                + (count_1 - 1) * results[1][cohort_name][col]["variance"]
            ) / dof_denominator

            # Check if pooled variance is valid (not NaN or Inf)
            if not np.isfinite(Sp):
                info(f"Skipping {col} in {cohort_name}: pooled variance is not finite")
                continue

            # Compute t-score denominator
            t_denominator = (
                (Sp / count_0) + (Sp / count_1)
            ) ** 0.5
            
            if t_denominator == 0 or not np.isfinite(t_denominator):
                info(f"Skipping {col} in {cohort_name}: t-score denominator is zero or not finite")
                continue

            # t value
            final_result[cohort_name][col] = {}
            t_score = (
                results[0][cohort_name][col]["average"]
                - results[1][cohort_name][col]["average"]
            ) / t_denominator

            dof = dof_denominator
            p_value = float(2 * (1 - t.cdf(np.abs(t_score), dof)))

            final_result[cohort_name][col] = {"p_value": p_value, "t_score": t_score}

    # return the final results of the algorithm
    return final_result


@federated
@dataframes
def t_test_partial(dataframes: dict[str, pd.DataFrame]) -> dict:
    results = {}
    for cohort_name, df in dataframes.items():
        results[cohort_name] = _t_test_partial(df)
    return results
    
def _t_test_partial(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """
    Compute the mean and the sample variance of a column for a single data station to
    share with the aggregator part of the algorithm

    Parameters
    ----------
    df : pd.DataFrame
        The data for the data station
    columns : list[str] | None
        The columns to compute the mean and sample variance for. The columns must be
        numeric. If not provided, all numeric columns are included.

    Returns
    -------
    dict
        The mean, the number of observations and the sample variance for the data
        station.
    """

    info("Checking number of records in the DataFrame.")
    MINIMUM_NUMBER_OF_RECORDS = get_env_var(
        "T_TEST_MINIMUM_NUMBER_OF_RECORDS",
        T_TEST_MINIMUM_NUMBER_OF_RECORDS,
        as_type="int",
    )
    if len(df) <= MINIMUM_NUMBER_OF_RECORDS:
        raise InputError(
            "Number of records in 'df' must be greater than "
            f"{MINIMUM_NUMBER_OF_RECORDS}."
        )

    if not columns:
        columns = df.select_dtypes(include=["number"]).columns.tolist()
    else:
        # Check that column names exist in the dataframe
        non_existing_columns = [col for col in columns if col not in df.columns]
        if non_existing_columns:
            raise InputError(
                f"Columns {non_existing_columns} do not exist in the dataframe"
            )
        # Check that columns are numerical
        non_numeric_columns = [
            col for col in columns if not ptypes.is_numeric_dtype(df[col])
        ]
        if non_numeric_columns:
            raise InputError(f"Columns {non_numeric_columns} are not numeric")

    # Compute mean and sample variance
    partial_results = {}

    for col in columns:
        # Mean and total count (N)
        info(f"Computing mean for {col}")
        count = df[col].count()
        # Check if count is not equal to 0 or 1 to avoid division by 0
        if count == 0 or count == 1:
            info(f"Skipping {col} due to insufficient data.")
            continue
        column_sum = df[col].sum()
        average = column_sum / count
        # Sample variance
        info(f"Computing sample variance for {col}")
        ssd = ((df[col].astype(float) - average) ** 2).sum()
        variance = ssd / (count - 1)

        partial_results[col] = {
            "average": float(average),
            "count": float(count),
            "variance": float(variance),
        }

    return partial_results
