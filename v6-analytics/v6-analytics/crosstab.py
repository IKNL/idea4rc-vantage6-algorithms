import pandas as pd

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import data
from vantage6.algorithm.tools.util import get_env_var
from vantage6.algorithm.tools.exceptions import (
    EnvironmentVariableError,
    PrivacyThresholdViolation,
)
from typing import Any
from io import StringIO
import pandas as pd
import scipy

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.decorator import algorithm_client, dataframes
from vantage6.algorithm.client import AlgorithmClient


@dataframes
def partial_crosstab(
    dataframes: dict[str, pd.DataFrame],
    results_col: str,
    group_cols: list[str],
) -> str:
    dfs = dataframes.values()
    cohort_names = dataframes.keys()
    results = {}
    for df, name in zip(dfs, cohort_names):
        results[name] = _partial_crosstab(df, results_col, group_cols)
    return results


@algorithm_client
def crosstab(
    client: AlgorithmClient,
    results_col: str,
    group_cols: list[str],
    organizations_to_include: list[int] = None,
    include_chi2: bool = True,
    include_totals: bool = True,
):
    """
    Central part of the algorithm

    Parameters
    ----------
    client : AlgorithmClient
        The client object used for communication with the server.
    results_col : str
        The column for which counts are calculated
    group_cols : list[str]
        List of one or more columns to group the data by.
    organizations_to_include : list[int], optional
        List of organization ids to include in the computation. If not provided, all
        organizations in the collaboration are included.
    include_chi2 : bool, optional
        Whether to include the chi-squared statistic in the results.
    include_totals : bool, optional
        Whether to include totals in the contingency table.
    """
    # get all organizations (ids) within the collaboration so you can send a
    # task to them.
    if not organizations_to_include:
        organizations = client.organization.list()
        organizations_to_include = [
            organization.get("id") for organization in organizations
        ]

    # Define input parameters for a subtask
    info("Defining input parameters")
    input_ = {
        "method": "partial_crosstab",
        "kwargs": {
            "results_col": results_col,
            "group_cols": group_cols,
        },
    }

    # create a subtask for all organizations in the collaboration.
    info("Creating subtask to compute partial contingency tables")
    task = client.task.create(
        input_=input_,
        organizations=organizations_to_include,
        name="Partial crosstabulation",
        description="Contingency table for each organization",
    )

    # wait for node to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")

    all_cohort_results = {}
    cohort_names = results[0].keys()
    for cohort_name in cohort_names:
        cohort_results = [result[cohort_name] for result in results]
        all_cohort_results[cohort_name] = _aggregate_results(
            cohort_results, group_cols, include_chi2, include_totals
        )

    # return the final results of the algorithm
    return all_cohort_results


#
# Original code
#


# The following global variables are algorithm settings. They can be overwritten by
# the node admin by setting the corresponding environment variables.

# Minimum value to be given as individual value in the contingency table. To be
# overwritten by setting the "CROSSTAB_PRIVACY_THRESHOLD" environment variable.
DEFAULT_PRIVACY_THRESHOLD = "0"

# Minimum number of rows in the node's dataset. To be overwritten by setting the
# "CROSSTAB_MINIMUM_ROWS_TOTAL" environment variable.
DEFAULT_MINIMUM_ROWS_TOTAL = "3"

# Whether or not to allow value of 0 in the contingency table. To be overwritten by
# setting the "CROSSTAB_ALLOW_ZERO" environment variable.
DEFAULT_ALLOW_ZERO = "true"


# @algorithm_client
# def central_crosstab(
#     client: AlgorithmClient,
#     results_col: str,
#     group_cols: list[str],
#     organizations_to_include: list[int] = None,
#     include_chi2: bool = True,
#     include_totals: bool = True,
# ) -> Any:
#     """
#     Central part of the algorithm

#     Parameters
#     ----------
#     client : AlgorithmClient
#         The client object used for communication with the server.
#     results_col : str
#         The column for which counts are calculated
#     group_cols : list[str]
#         List of one or more columns to group the data by.
#     organizations_to_include : list[int], optional
#         List of organization ids to include in the computation. If not provided, all
#         organizations in the collaboration are included.
#     include_chi2 : bool, optional
#         Whether to include the chi-squared statistic in the results.
#     include_totals : bool, optional
#         Whether to include totals in the contingency table.
#     """
#     # get all organizations (ids) within the collaboration so you can send a
#     # task to them.
#     if not organizations_to_include:
#         organizations = client.organization.list()
#         organizations_to_include = [
#             organization.get("id") for organization in organizations
#         ]

#     # Define input parameters for a subtask
#     info("Defining input parameters")
#     input_ = {
#         "method": "partial_crosstab",
#         "kwargs": {
#             "results_col": results_col,
#             "group_cols": group_cols,
#         },
#     }

#     # create a subtask for all organizations in the collaboration.
#     info("Creating subtask to compute partial contingency tables")
#     task = client.task.create(
#         input_=input_,
#         organizations=organizations_to_include,
#         name="Partial crosstabulation",
#         description="Contingency table for each organization",
#     )

#     # wait for node to return results of the subtask.
#     info("Waiting for results")
#     results = client.wait_for_results(task_id=task.get("id"))
#     info("Results obtained!")

#     # return the final results of the algorithm
#     return _aggregate_results(results, group_cols, include_chi2, include_totals)


def _aggregate_results(
    results: dict, group_cols: list[str], include_chi2: bool, include_totals: bool
) -> pd.DataFrame:
    """
    Aggregate the results of the partial computations.

    Parameters
    ----------
    results : list[dict]
        List of JSON data containing the partial results.
    group_cols : list[str]
        List of columns that were used to group the data.
    include_chi2 : bool
        Whether to include the chi-squared statistic in the results.
    include_totals : bool
        Whether to include totals in the contingency table.

    Returns
    -------
    pd.DataFrame
        Aggregated results.
    """
    # The results are pandas dictionaries converted to JSON. Convert them back and
    # then add them together to get the final partial_df.
    partial_dfs = []
    for result in results:
        df = pd.read_json(StringIO(result))
        # set group cols as index
        df.set_index(group_cols, inplace=True)
        partial_dfs.append(df)

    # Get all unique values for the result column
    all_result_levels = list(set([col for df in partial_dfs for col in df.columns]))

    # The partial results are already in the form of a contingency table, but they
    # contain ranges (e.g. "0-5"). These are converted to two columns: one for the
    # minimum value and one for the maximum value.
    converted_results = []
    all_orig_columns = set()
    for partial_df in partial_dfs:
        # expand the ranges to min and max values
        orig_columns = partial_df.columns
        all_orig_columns.update(orig_columns)
        for col in orig_columns:
            if partial_df[col].dtype == "object":
                # if the column contains a range, split it into two columns
                partial_df[[f"{col}_min", f"{col}_max"]] = partial_df[col].str.split(
                    "-", expand=True
                )
                partial_df[f"{col}_max"] = partial_df[f"{col}_max"].fillna(
                    partial_df[f"{col}_min"]
                )
            else:
                # column is already numeric: simply copy it to the new columns
                partial_df[f"{col}_min"] = partial_df[col]
                partial_df[f"{col}_max"] = partial_df[col]
        # drop the original columns
        partial_df.drop(columns=orig_columns, inplace=True)
        # convert to numeric
        partial_df = partial_df.apply(pd.to_numeric).astype(int)
        converted_results.append(partial_df)

    orig_columns = list(all_orig_columns)

    # We now have a list of partial results that contain minimum and maximum values
    # for each cell in the contingency table. We can now add them together to get the
    # final result.
    aggregated_df = pd.concat(converted_results).fillna(0).astype(int)
    aggregated_df = aggregated_df.groupby(aggregated_df.index).sum()

    # above groupby puts multiindex groupby into tuples, which we need to unpack
    if len(group_cols) > 1:
        aggregated_df.index = pd.MultiIndex.from_tuples(
            aggregated_df.index, names=group_cols
        )

    min_colnames = [f"{col}_min" for col in orig_columns]
    max_colnames = [f"{col}_max" for col in orig_columns]
    # Compute chi-squared statistic
    if include_chi2:
        chi2, chi2_pvalue = compute_chi_squared(
            aggregated_df, min_colnames, max_colnames
        )

    if include_totals:
        col_totals, row_totals, total_total = _compute_totals(
            aggregated_df, min_colnames, max_colnames, orig_columns
        )

    # Convert back to strings so that we can add ranges
    aggregated_df = aggregated_df.astype(str)

    # Finally, we need to combine the min and max values back into ranges
    min_max_cols = aggregated_df.columns
    for level in all_result_levels:
        aggregated_df[level] = _concatenate_min_max(
            aggregated_df[f"{level}_min"], aggregated_df[f"{level}_max"]
        )

    # clean up: drop the min and max columns
    aggregated_df.drop(min_max_cols, axis=1, inplace=True)

    # reset index to pass the group columns along to results
    aggregated_df = aggregated_df.reset_index()

    # add totals ranges
    if include_totals:
        # ensure columns are ordered same in totals as in the rest of the table
        column_order = aggregated_df.columns[len(group_cols) :]
        col_totals = col_totals.reindex(column_order)
        # add totals
        aggregated_df["Total"] = row_totals
        aggregated_df.loc[len(aggregated_df)] = (
            ["Total"]
            + ["" for _ in group_cols[1:]]
            + col_totals.tolist()
            + [total_total]
        )

    results = {"contingency_table": aggregated_df.to_dict(orient="records")}
    if include_chi2:
        results.update({"chi2": {"chi2": chi2, "P-value": chi2_pvalue}})
    return results


def compute_chi_squared(
    contingency_table: pd.DataFrame, min_colnames: list[str], max_colnames: list[str]
) -> tuple[str]:
    """
    Compute chi squared statistic based on the contingency table

    Parameters
    ----------
    contingency_table : pd.DataFrame
        The contingency table.
    min_colnames : list[str]
        List of column names for the minimum values of each range.
    max_colnames : list[str]
        List of column names for the maximum values of each range.

    Returns
    -------
    tuple[str]
        Tuple containing the chi-squared statistics and pvalues. If the contingency
        table contains ranges, the statistics and pvalues are also returned as a range.
    """
    info("Computing chi-squared statistic...")

    # for minimum values, remove rows/columns with only zeros
    min_df = contingency_table[min_colnames]
    min_df = min_df.loc[(min_df != 0).any(axis=1)]
    min_df = min_df.loc[:, (min_df != 0).any(axis=0)]
    max_df = contingency_table[max_colnames]
    max_df = min_df.loc[(max_df != 0).any(axis=1)]
    max_df = min_df.loc[:, (max_df != 0).any(axis=0)]

    chi2_min = scipy.stats.chi2_contingency(min_df)
    chi2_max = scipy.stats.chi2_contingency(max_df)

    if chi2_min.statistic == chi2_max.statistic:
        return str(chi2_min.statistic), str(chi2_min.pvalue)

    # note that if giving a range, MAX goes before MIN, because the values of the
    # statistic are actually larger for the minimum side of the range, because then the
    # difference with the average is larger than for the maximum side of the range
    # (p-value is not reversed as that is again the reverse of the statistic :-))
    return (
        f"{chi2_max.statistic} - {chi2_min.statistic}",
        f"{chi2_min.pvalue} - {chi2_max.pvalue}",
    )


def _compute_totals(
    contingency_table: pd.DataFrame,
    min_colnames: list[str],
    max_colnames: list[str],
    orig_columns: list[str],
) -> tuple:
    """
    Compute the totals for the contingency table.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        The contingency table.
    min_colnames : list[str]
        List of column names for the minimum values of each range.
    max_colnames : list[str]
        List of column names for the maximum values of each range.
    orig_columns : list[str]
        List of original column names.

    Returns
    -------
    tuple
        Tuple containing the column totals, row totals, and the sum of all data points.
    """
    # Add totals
    min_row_totals = contingency_table[min_colnames].sum(axis=1)
    min_col_totals = contingency_table[min_colnames].sum(axis=0)
    max_row_totals = contingency_table[max_colnames].sum(axis=1)
    max_col_totals = contingency_table[max_colnames].sum(axis=0)

    min_total_total = min_row_totals.sum()
    max_total_total = max_row_totals.sum()
    if min_total_total != max_total_total:
        total_total = f"{min_total_total} - {max_total_total}"
    else:
        total_total = str(min_total_total)

    min_row_totals = min_row_totals.astype(str).reset_index(drop=True)
    min_col_totals = min_col_totals.astype(str).reset_index(drop=True)
    max_row_totals = max_row_totals.astype(str).reset_index(drop=True)
    max_col_totals = max_col_totals.astype(str).reset_index(drop=True)

    # check if the totals are the same
    col_totals = _concatenate_min_max(min_col_totals, max_col_totals)
    col_totals.index = orig_columns
    row_totals = _concatenate_min_max(min_row_totals, max_row_totals)
    return col_totals, row_totals, total_total


def _concatenate_min_max(min_col: pd.Series, max_col: pd.Series) -> pd.Series:
    """
    Concatenate two columns into a single column with ranges.

    Parameters
    ----------
    min_col : pd.Series
        The column with minimum values.
    max_col : pd.Series
        The column with maximum values.

    Returns
    -------
    pd.Series
        The concatenated column.
    """
    # first, simply concatenate the columns
    result = min_col + "-" + max_col
    # then, remove extra text where min and max are the same
    min_max_same = min_col == max_col
    result.loc[min_max_same] = result.loc[min_max_same].str.replace(
        r"-(\d+)", "", regex=True
    )
    return result


# TODO create PR at v6-crosstab-py to add this function
def _partial_crosstab(
    df: pd.DataFrame,
    results_col: str,
    group_cols: list[str],
) -> str:
    """
    Decentral part of the algorithm

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    results_col : str
        The column for which counts are calculated
    group_cols : list[str]
        List of one or more columns to group the data by.

    Returns
    -------
    str
        The contingency table as a JSON string.

    Raises
    ------
    PrivacyThresholdViolation
        The privacy threshold is not met by any values in the contingency table.
    """
    # get environment variables with privacy settings
    # pylint: disable=invalid-name
    PRIVACY_THRESHOLD = _convert_envvar_to_int(
        "CROSSTAB_PRIVACY_THRESHOLD", DEFAULT_PRIVACY_THRESHOLD
    )
    ALLOW_ZERO = _convert_envvar_to_bool("CROSSTAB_ALLOW_ZERO", DEFAULT_ALLOW_ZERO)

    # check if env var values are compatible
    info("Checking privacy settings before starting...")
    _do_prestart_privacy_checks(
        df, group_cols + [results_col], PRIVACY_THRESHOLD, ALLOW_ZERO
    )

    # TODO this is a fix for categorical columns with empty values.
    for col in df.select_dtypes(include=["category"]).columns:
        if "N/A" not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories("N/A")

    # Fill empty (categorical) values with "N/A"
    df = df.fillna("N/A")

    # Create contingency table
    info("Creating contingency table...")
    cross_tab_df = (
        df.groupby(group_cols + [results_col], dropna=False)[results_col]
        .count()
        .unstack(level=results_col)
        .fillna(0)
        .astype(int)
    )
    info("Contingency table created!")

    # if no values are higher than the threshold, return an error. But before doing so,
    # filter out the N/A values: if a column is requested that contains only unique
    # values but also empty values, the crosstab would otherwise share unique values as
    # categories if there are enough empty values to meet the threshold
    info("Checking if privacy threshold is met by any values...")
    non_na_crosstab_df = cross_tab_df.drop(columns="N/A", errors="ignore")
    for col in group_cols:
        non_na_crosstab_df = non_na_crosstab_df.iloc[
            non_na_crosstab_df.index.get_level_values(col) != "N/A"
        ]
    if not (non_na_crosstab_df >= PRIVACY_THRESHOLD).any().any():
        raise PrivacyThresholdViolation(
            "No values in the contingency table are higher than the privacy threshold "
            f"of {PRIVACY_THRESHOLD}. Please check if you submitted categorical "
            "variables - if you did, there may simply not be enough data at this node."
        )

    # Replace too low values with a privacy-preserving value
    info("Replacing values below threshold with privacy-enhancing values...")
    replace_value = 1 if ALLOW_ZERO else 0
    replace_condition = (
        (cross_tab_df >= PRIVACY_THRESHOLD) | (cross_tab_df == 0)
        if ALLOW_ZERO
        else (cross_tab_df >= PRIVACY_THRESHOLD)
    )
    cross_tab_df.where(replace_condition, replace_value, inplace=True)

    # Cast to string and set non-privacy-preserving values to a placeholder indicating
    # that the value is below the threshold
    BELOW_THRESHOLD_PLACEHOLDER = _get_threshold_placeholder(
        PRIVACY_THRESHOLD, ALLOW_ZERO
    )
    cross_tab_df = cross_tab_df.astype(str).where(
        replace_condition, BELOW_THRESHOLD_PLACEHOLDER
    )

    # reset index to ensure that groups are passed along to central part
    cross_tab_df = cross_tab_df.reset_index()

    # Cast results to string to ensure they can be read again
    info("Returning results!")
    return cross_tab_df.astype(str).to_json(orient="records")


def _do_prestart_privacy_checks(
    df: pd.DataFrame,
    requested_columns: list[str],
    privacy_threshold: int,
    allow_zero: bool,
) -> None:
    """
    Perform privacy checks before starting the computation.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    requested_columns : list[str]
        The columns requested for the computation.
    privacy_threshold : int
        The privacy threshold value.
    allow_zero : bool
        The flag indicating whether zero values are allowed.

    Raises
    ------
    EnvironmentVariableError
        The environment variables set by the node are not compatible.

    """
    minimum_rows_total = _convert_envvar_to_int(
        "CROSSTAB_MINIMUM_ROWS_TOTAL", DEFAULT_MINIMUM_ROWS_TOTAL
    )

    if privacy_threshold == 0 and not allow_zero:
        raise EnvironmentVariableError(
            "Privacy threshold is set to 0, but zero values are not allowed. This "
            "directly contradicts each other - please change one of the settings."
        )

    # Check if dataframe contains enough rows
    if len(df) < minimum_rows_total:
        raise PrivacyThresholdViolation(
            f"Dataframe contains less than {minimum_rows_total} rows. Refusing to "
            "handle this computation, as it may lead to privacy issues."
        )

    # Check if requested columns are allowed
    allowed_columns = get_env_var("CROSSTAB_ALLOWED_COLUMNS")
    if allowed_columns:
        allowed_columns = allowed_columns.split(",")
        for col in requested_columns:
            if col not in allowed_columns:
                raise ValueError(
                    f"The node administrator does not allow '{col}' to be requested in "
                    "this algorithm computation. Please contact the node administrator "
                    "for more information."
                )
    non_allowed_collumns = get_env_var("CROSSTAB_DISALLOWED_COLUMNS")
    if non_allowed_collumns:
        non_allowed_collumns = non_allowed_collumns.split(",")
        for col in requested_columns:
            if col in non_allowed_collumns:
                raise ValueError(
                    f"The node administrator does not allow '{col}' to be requested in "
                    "this algorithm computation. Please contact the node administrator "
                    "for more information."
                )


def _get_threshold_placeholder(privacy_threshold: int, allow_zero: bool) -> str:
    """
    Get the below threshold placeholder based on the privacy threshold and allow zero flag.

    Parameters
    ----------
    privacy_threshold : int
        The privacy threshold value.
    allow_zero : bool
        The flag indicating whether zero values are allowed.

    Returns
    -------
    str
        The below threshold placeholder.
    """
    if allow_zero:
        if privacy_threshold > 2:
            return f"1-{privacy_threshold-1}"
        else:
            return "1"
    else:
        if privacy_threshold > 1:
            return f"0-{privacy_threshold-1}"
        else:
            return "0"


def _convert_envvar_to_bool(envvar_name: str, default: str) -> bool:
    """
    Convert an environment variable to a boolean value.

    Parameters
    ----------
    envvar_name : str
        The environment variable name to convert.
    default : str
        The default value to use if the environment variable is not set.

    Returns
    -------
    bool
        The boolean value of the environment variable.
    """
    envvar = get_env_var(envvar_name, default).lower()
    if envvar in ["true", "1", "yes", "t"]:
        return True
    elif envvar in ["false", "0", "no", "f"]:
        return False
    else:
        raise ValueError(
            f"Environment variable '{envvar_name}' has value '{envvar}' which cannot be"
            " converted to a boolean value. Please use 'false' or 'true'."
        )


def _convert_envvar_to_int(envvar_name: str, default: str) -> int:
    """
    Convert an environment variable to an integer value.

    Parameters
    ----------
    envvar_name : str
        The environment variable name to convert.
    default : str
        The default value to use if the environment variable is not set.

    Returns
    -------
    int
        The integer value of the environment variable.
    """
    envvar = get_env_var(envvar_name, default)
    error_msg = (
        f"Environment variable '{envvar_name}' has value '{envvar}' which cannot be "
        "converted to a positive integer value."
    )
    try:
        envvar = int(envvar)
    except ValueError as exc:
        raise ValueError(error_msg) from exc
    if envvar < 0:
        raise ValueError(error_msg)
    return envvar
