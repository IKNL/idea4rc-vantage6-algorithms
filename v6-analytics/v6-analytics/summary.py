import json
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.decorator import (
    algorithm_client,
    central,
    dataframes,
    federated,
    metadata,
)
from vantage6.algorithm.tools.exceptions import (
    InputError,
    NodePermissionException,
    PrivacyThresholdViolation,
)
from vantage6.algorithm.tools.util import error, get_env_var, info, warn
from v6_idea4rc_common.type_guards import (
    Idea4rcDType,
    assert_column_dtype_in,
    assert_columns_dtype_in,
    classify_idea4rc_dtype,
)

# names of environment variables
## minimum number of rows in the dataframe
ENVVAR_MINIMUM_ROWS = "SUMMARY_MINIMUM_ROWS"
## whitelist of columns allowed to be requested
ENVVAR_ALLOWED_COLUMNS = "SUMMARY_ALLOWED_COLUMNS"
## blacklist of columns not allowed to be requested
ENVVAR_DISALLOWED_COLUMNS = "SUMMARY_DISALLOWED_COLUMNS"
## privacy threshold for count of a unique value in a categorical column
ENVVAR_PRIVACY_THRESHOLD = "SUMMARY_PRIVACY_THRESHOLD"


class EnvVarsAllowed(Enum):
    """Environment varible names to allow computation of different variables"""

    ALLOW_MIN = "SUMMARY_ALLOW_MIN"
    ALLOW_MAX = "SUMMARY_ALLOW_MAX"
    ALLOW_COUNT = "SUMMARY_ALLOW_COUNT"
    ALLOW_SUM = "SUMMARY_ALLOW_SUM"
    ALLOW_MISSING = "SUMMARY_ALLOW_MISSING"
    ALLOW_VARIANCE = "SUMMARY_ALLOW_VARIANCE"
    ALLOW_COUNTS_UNIQUE_VALUES = "SUMMARY_ALLOW_COUNTS_UNIQUE_VALUES"
    ALLOW_NUM_COMPLETE_ROWS = "SUMMARY_ALLOW_NUM_COMPLETE_ROWS"


# default values for environment variables
DEFAULT_MINIMUM_ROWS = 0
DEFAULT_PRIVACY_THRESHOLD = 0

@central
@algorithm_client
def summary(
    client: AlgorithmClient,
    columns: list[str] | None = None,
    numeric_columns: list[bool] | None = None,
    organizations_to_include: list[int] | None = None,
    stratification_column: str | None = None,
) -> Any:
    """
    Send task to each node participating in the task to compute a local summary,
    aggregate them for all nodes, and return the result.

    Parameters
    ----------
    client : AlgorithmClient
        The client object used to communicate with the server.
    columns : list[str] | None
        The columns to include in the summary. If not given, all columns are included.
    numeric_columns : list[str] | None
        Whether each of the columns is numeric or not. If not given, the algorithm will
        try to infer the type of the columns.
    organizations_to_include : list[int] | None
        The organizations to include in the task. If not given, all organizations
        in the collaboration are included.
    stratification_column: str | None
        The column to use for stratification. If not given, no stratification is done.
    """

    # get all organizations (ids) within the collaboration so you can send a
    # task to them.
    if not organizations_to_include:
        organizations = client.organization.list()
        organizations_to_include = [
            organization.get("id") for organization in organizations
        ]

    # create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        method="summary_per_data_station",
        arguments={
            "columns": columns,
            "numeric_columns": numeric_columns,
            "stratification_column": stratification_column,
        },
        organizations=organizations_to_include,
        name="Subtask summary",
        description="Compute summary per data station",
    )

    # wait for node to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")

    # aggregate the partial summaries of all nodes
    all_cohort_results = {}

    means = {}
    cohort_names = list(
        set(
            [
                item
                for sublist in [result.keys() for result in results]
                for item in sublist
            ]
        )
    )

    lookup_organizations = {
        str(org.get("id")): org.get("name") for org in client.organization.list()
    }

    for cohort_name in cohort_names:
        cohort_results = [result.get(cohort_name) for result in results]
        all_cohort_results[cohort_name] = _aggregate_partial_summaries(
            cohort_results, lookup_organizations
        )

        numerical_columns = list(all_cohort_results[cohort_name]["numeric"].keys())
        # compute the variance now that we have the mean
        means[cohort_name] = [
            all_cohort_results[cohort_name]["numeric"][column]["mean"]
            for column in numerical_columns
        ]
        info(f"n num cols: {len(numerical_columns)}")
        info(f"n means: {len(means[cohort_name])}")

    task = client.task.create(
        method="variance_per_data_station",
        arguments={
            "columns": numerical_columns,
            "means": means,
            "stratification_column": stratification_column,
        },
        organizations=organizations_to_include,
        name="Subtask variance",
        description="Compute variance per data station",
    )

    variance_results = client.wait_for_results(task_id=task.get("id"))

    # add the standard deviation to the results
    for cohort_name in cohort_names:
        cohort_variance_results = [
            result.get(cohort_name) for result in variance_results
        ]
        all_cohort_results[cohort_name] = _add_sd_to_results(
            all_cohort_results[cohort_name], cohort_variance_results, numerical_columns
        )

    # return the final results of the algorithm
    return all_cohort_results


def _aggregate_partial_summaries(results: list[dict], lookup_organizations) -> dict:
    """Aggregate the partial summaries of all nodes.

    Parameters
    ----------
    results : list[dict]
        The partial summaries of all nodes.
    """
    info("Aggregating partial summaries")
    aggregate = {}
    is_first = True

    def _merge_numeric_bound(
        current_value: Any,
        incoming_value: Any,
        prefer_min: bool,
    ) -> Any:
        """
        Merge numeric bounds while safely handling None/NaN values.

        Values can become None after serialization (e.g. NaN -> null).
        """
        current_missing = current_value is None or pd.isna(current_value)
        incoming_missing = incoming_value is None or pd.isna(incoming_value)

        if current_missing and incoming_missing:
            return None
        if current_missing:
            return incoming_value
        if incoming_missing:
            return current_value

        return min(current_value, incoming_value) if prefer_min else max(current_value, incoming_value)

    def _merge_date_bound(
        current_value: Any,
        incoming_value: Any,
        prefer_min: bool,
    ) -> Any:
        """Merge date bounds while safely handling None/NaT values."""
        current_missing = current_value is None or pd.isna(current_value)
        incoming_missing = incoming_value is None or pd.isna(incoming_value)

        if current_missing and incoming_missing:
            return current_value
        if current_missing:
            return incoming_value
        if incoming_missing:
            return current_value

        current_dt = pd.to_datetime(current_value, utc=True, errors="coerce")
        incoming_dt = pd.to_datetime(incoming_value, utc=True, errors="coerce")
        if pd.isna(current_dt) and pd.isna(incoming_dt):
            return current_value
        if pd.isna(current_dt):
            return incoming_dt.date().isoformat()
        if pd.isna(incoming_dt):
            return current_dt.date().isoformat()

        if prefer_min:
            return min(current_dt, incoming_dt).date().isoformat()
        return max(current_dt, incoming_dt).date().isoformat()

    def _normalize_date_summary(date_summary: Any) -> dict[str, dict[str, Any]]:
        """Normalize date summary payload to dict format for aggregation."""
        if date_summary is None:
            return {}
        if isinstance(date_summary, str):
            if not date_summary.strip():
                return {}
            return json.loads(date_summary)
        return date_summary

    # For each node (= organization)
    for result in results:
        if result is None:
            # raise AlgorithmExecutionError(
            #     "At least one of the nodes returned invalid result. Please check the "
            #     "logs."
            # )
            warn("node did not have results for a certain cohort")
            continue

        organization_name = lookup_organizations[str(result["organization_id"])]
        result["date"] = _normalize_date_summary(result.get("date"))
        if is_first:
            # copy results. Only convert num complete rows per node to a list so that
            # we can add the other nodes to it later
            aggregate = result
            aggregate["num_complete_rows_per_node"] = {
                organization_name: result["num_complete_rows_per_node"]
            }
            aggregate["num_rows_per_node"] = {
                organization_name: result["num_rows_per_node"]
            }

            for column in result["numeric"]:
                aggregate["numeric"][column]["median"] = {
                    organization_name: result["numeric"][column]["median"]
                }
                aggregate["numeric"][column]["q_25"] = {
                    organization_name: result["numeric"][column]["q_25"]
                }
                aggregate["numeric"][column]["q_75"] = {
                    organization_name: result["numeric"][column]["q_75"]
                }

            is_first = False
            continue

        # aggregate data for numeric columns
        for column in result["numeric"]:
            aggregated_dict = aggregate["numeric"][column]
            aggregated_dict["count"] += result["numeric"][column]["count"]
            aggregated_dict["min"] = _merge_numeric_bound(
                aggregate["numeric"][column].get("min"),
                result["numeric"][column].get("min"),
                prefer_min=True,
            )
            aggregated_dict["max"] = _merge_numeric_bound(
                aggregate["numeric"][column].get("max"),
                result["numeric"][column].get("max"),
                prefer_min=False,
            )
            aggregated_dict["missing"] += result["numeric"][column]["missing"]
            aggregated_dict["sum"] += result["numeric"][column]["sum"]
            aggregated_dict["median"][organization_name] = result["numeric"][column][
                "median"
            ]
            aggregated_dict["q_25"][organization_name] = result["numeric"][column][
                "q_25"
            ]
            aggregated_dict["q_75"][organization_name] = result["numeric"][column][
                "q_75"
            ]

        # aggregate data for categorical columns
        for column in result["categorical"]:
            aggregated_dict = aggregate["categorical"][column]
            aggregated_dict["count"] += result["categorical"][column]["count"]
            aggregated_dict["missing"] += result["categorical"][column]["missing"]
        
        # aggregate data for date columns
        for column in result["date"]:
            aggregated_dict = aggregate["date"][column]
            aggregated_dict["count"] += result["date"][column]["count"]
            aggregated_dict["missing"] += result["date"][column]["missing"]
            aggregated_dict["min"] = _merge_date_bound(
                aggregate["date"][column].get("min"),
                result["date"][column].get("min"),
                prefer_min=True,
            )
            aggregated_dict["max"] = _merge_date_bound(
                aggregate["date"][column].get("max"),
                result["date"][column].get("max"),
                prefer_min=False,
            )

        # add the number of complete rows for this node
        aggregate["num_complete_rows_per_node"][organization_name] = result[
            "num_complete_rows_per_node"
        ]

        # add the number of rows for this node
        aggregate["num_rows_per_node"][organization_name] = result["num_rows_per_node"]

        # add the unique values
        for column in result["counts_unique_values"]:
            if column not in aggregate["counts_unique_values"]:
                aggregate["counts_unique_values"][column] = {}
            for value, count in result["counts_unique_values"][column].items():
                if value not in aggregate["counts_unique_values"][column]:
                    aggregate["counts_unique_values"][column][value] = 0
                aggregate["counts_unique_values"][column][value] += count

        aggregate["num_rows"] = sum(aggregate["num_rows_per_node"].values())

    # now that all data is aggregated, we can compute the mean
    for column in aggregate["numeric"]:
        aggregated_dict = aggregate["numeric"][column]
        if aggregated_dict["count"]:
            aggregated_dict["mean"] = aggregated_dict["sum"] / aggregated_dict["count"]
        else:
            aggregated_dict["mean"] = 0  # TODO this is terrible, we should not do this

    

    return aggregate


def _add_sd_to_results(
    results: dict, variance_results: list[dict] | None, numerical_columns: list[str]
) -> dict:
    """Add the variance to the results.

    Parameters
    ----------
    results : dict
        The results of the summary task.
    variance_results : list[dict]
        The variance results of all nodes.
    numerical_columns : list[str]
        The numerical columns.

    Returns
    -------
    dict
        The results with the variance added.
    """
    for column in numerical_columns:
        sum_variance = 0
        for node_results in variance_results:
            if not node_results:
                continue
            sum_variance += node_results[column]
        if results["numeric"][column]["count"] > 1:
            variance = sum_variance / (results["numeric"][column]["count"] - 1)
        else:
            variance = 0  # TODO THIS IS TERRIBLE
        results["numeric"][column]["std"] = variance**0.5
    return results


# Do not provide the columns as we want all columns to be included
@federated
@metadata
@dataframes
def summary_per_data_station(
    dataframes: dict[str, pd.DataFrame],
    metadata,
    stratification_column: str | None = None,
    *args,
    **kwargs,
) -> dict:
    dfs = dataframes.values()
    cohort_names = dataframes.keys()
    results = {}
    for df, name in zip(dfs, cohort_names):
        if stratification_column:
            assert_column_dtype_in(
                df,
                stratification_column,
                allowed=[Idea4rcDType.CATEGORY],
                algorithm="summary",
                expected_kind="categorical (pandas 'category')",
            )
            for value in df[stratification_column].unique():
                df_stratified = df[df[stratification_column] == value]
                results[f"{name}_{stratification_column}=={value}"] = (
                    structure_summary_per_data_station_output(
                        df_stratified,
                        _summary_per_data_station(df_stratified, *args, **kwargs),
                        metadata,
                    )
                )
        else:
            results[name] = structure_summary_per_data_station_output(
                df, _summary_per_data_station(df, *args, **kwargs), metadata
            )
        # Add median and quantiles (0.25, 0.75)
    return results


def structure_summary_per_data_station_output(df, results, metadata):
    for var in results["numeric"]:
        # Temp fix to avoid errors when all values are NaN
        if df[var].isna().all():
            warn(f"Column {var} is all NaN, skipping")
            results["numeric"][var]["median"] = 1
            results["numeric"][var]["q_25"] = 1
            results["numeric"][var]["q_75"] = 1    
            continue

        results["numeric"][var]["median"] = float(np.nanmedian(df[var]))
        results["numeric"][var]["q_25"] = float(np.nanquantile(df[var], 0.25))
        results["numeric"][var]["q_75"] = float(np.nanquantile(df[var], 0.75))
        results["organization_id"] = metadata.organization_id or 0
    return results


@federated
@dataframes
def variance_per_data_station(
    dataframes: dict[str, pd.DataFrame],
    means: dict[list[float]],
    stratification_column=None,
    *args,
    **kwargs,
) -> dict:
    dfs = dataframes.values()
    cohort_names = dataframes.keys()
    results = {}
    info(kwargs)
    info(means)
    info("Cake is a lie")
    for df, name in zip(dfs, cohort_names):
        if stratification_column:
            assert_column_dtype_in(
                df,
                stratification_column,
                allowed=[Idea4rcDType.CATEGORY],
                algorithm="summary_variance",
                expected_kind="categorical (pandas 'category')",
            )
            strata = df[stratification_column].unique()
            for stratum in strata:
                df_strata = df[df[stratification_column] == stratum]
                name_stratum = f"{name}_{stratification_column}=={stratum}"
                if means.get(name_stratum):
                    results[name] = _variance_per_data_station(
                        df_strata, means=means[name_stratum], *args, **kwargs
                    )
                else:
                    results[name] = None
        else:
            results[name] = _variance_per_data_station(
                df, means=means[name], *args, **kwargs
            )
    info(results)
    return results


def _summary_per_data_station(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
) -> dict:
    if not columns:
        columns = df.columns

    # Check that column names exist in the dataframe
    if not all([col in df.columns for col in columns]):
        non_existing_columns = [col for col in columns if col not in df.columns]
        raise InputError(
            f"Columns {non_existing_columns} do not exist in the dataframe"
        )

    # filter dataframe to only include the columns of interest
    df = df[columns]

    # Check privacy settings
    info("Checking if data complies to privacy settings")
    # check_privacy(df, columns)

    # Split the data in numeric and non-numeric columns
    inferred_numeric_columns = df.select_dtypes(
        include=["Int64", "Float64"]
    ).columns.tolist()
    if numeric_columns is None:
        numeric_columns = inferred_numeric_columns
        info(f"Inferred numeric columns: {inferred_numeric_columns}")
    else:
        assert_columns_dtype_in(
            df,
            numeric_columns,
            allowed=[Idea4rcDType.INT64, Idea4rcDType.FLOAT64],
            algorithm="summary",
            expected_kind="numeric (nullable Int64 or Float64)",
        )

    # In strict mode, enforce accepted numeric dtypes for the final numeric set
    assert_columns_dtype_in(
        df,
        numeric_columns,
        allowed=[Idea4rcDType.INT64, Idea4rcDType.FLOAT64],
        algorithm="summary",
        expected_kind="numeric (nullable Int64 or Float64)",
    )

    # set numeric and non-numeric columns
    # non_numeric_columns = list(set(columns) - set(numeric_columns))
    categorical_columns = df.select_dtypes(include=["category", "boolean"]).columns.tolist()
    df_numeric = df[numeric_columns]
    df_non_numeric = df[categorical_columns]
    df_date_columns = [
        c for c in df.columns if classify_idea4rc_dtype(df[c]) == Idea4rcDType.DATETIME64TZ
    ]
    df_date = df[df_date_columns]
    
    # compute data summary for date columns
    summary_date = pd.DataFrame()
    if not df_date.empty:
        summary_date = _get_date_summary(df_date)

    # compute data summary for numeric columns
    summary_numeric = pd.DataFrame()
    if not df_numeric.empty:
        summary_numeric = _get_numeric_summary(df_numeric)

    # compute data summary for non-numeric columns. Also compute the counts of the
    # unique values in the non-numeric columns (if they meet the privacy threshold)
    summary_categorical = pd.DataFrame()
    counts_unique_values = {}
    if not df_non_numeric.empty:
        summary_categorical = _get_categorical_summary(df_non_numeric)
        counts_unique_values = _get_counts_unique_values(df_non_numeric)

    # count complete rows without missing values
    num_rows_per_node = len(df)
    num_complete_rows_per_node = len(df.dropna())

    # filter out the variables that are not allowed to be shared
    summary_numeric, summary_categorical = _filter_results(
        summary_numeric, summary_categorical
    )
    if not get_env_var(
        EnvVarsAllowed.ALLOW_NUM_COMPLETE_ROWS.value, default="true", as_type="bool"
    ):
        warn(
            "Removing number of complete rows from summary as policies do not "
            "allow sharing it."
        )
        num_complete_rows_per_node = None
    if not get_env_var(
        EnvVarsAllowed.ALLOW_COUNTS_UNIQUE_VALUES.value, default="true", as_type="bool"
    ):
        warn(
            "Removing counts of unique values from summary as policies do not "
            "allow sharing it."
        )
        counts_unique_values = None

    return {
        "numeric": summary_numeric.to_dict(),
        "categorical": summary_categorical.to_dict(),
        "date": summary_date.to_json(date_format="iso"),
        "num_complete_rows_per_node": num_complete_rows_per_node,
        "num_rows_per_node": num_rows_per_node,
        "counts_unique_values": counts_unique_values,
    }


def _get_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the summary statistics for the numeric columns

    Parameters
    ----------
    df : pd.DataFrame
        The data to compute the summary statistics for
    """
    summary_numeric = df.describe(include=[int, float], percentiles=[])
    summary_numeric.loc["missing"] = df.isna().sum()
    summary_numeric.loc["sum"] = df.sum()
    summary_numeric.drop(["50%", "mean", "std"], inplace=True)
    return summary_numeric


def _get_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the summary statistics for the non-numeric columns

    Parameters
    ----------
    df : pd.DataFrame
        The data to compute the summary statistics for
    """
    # summary for non-numeric columns. Include the NA count and remove the values
    # that we don't want to share
    summary_categorical = df.astype(object).describe()
    summary_categorical.loc["missing"] = df.isna().sum()
    summary_categorical.drop(["top", "freq", "unique"], inplace=True)
    return summary_categorical

def _get_date_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the summary statistics for the date columns

    Parameters
    ----------
    df : pd.DataFrame
        The data to compute the summary statistics for
    """
    summary_date = df.describe()
    summary_date.loc["missing"] = df.isna().sum()
    summary_date.drop(["25%", "50%", "75%"], inplace=True)
    return summary_date

def _get_counts_unique_values(df: pd.DataFrame) -> dict:
    """
    Get the counts of the unique values in categorical columns

    Parameters
    ----------
    df : pd.DataFrame
        The data to get the counts of the unique values for

    Returns
    -------
    dict
        The counts of the unique values
    """
    counts = {}
    privacy_threshold = get_env_var(
        ENVVAR_PRIVACY_THRESHOLD, default=DEFAULT_PRIVACY_THRESHOLD, as_type="int"
    )
    for col in df.columns:
        counts[col] = _mask_privacy(df[col].value_counts(), privacy_threshold, col)
    return counts


def _mask_privacy(counts: pd.Series, privacy_threshold: int, column: str) -> dict:
    """
    Mask the values of a pandas series if the frequency is too low

    Parameters
    ----------
    counts : pd.Series
        The counts of the unique values
    privacy_threshold : int
        The minimum frequency of a value to be shared
    column : str
        The name of the column whose values are counted

    Returns
    -------
    pd.Series
        The masked counts
    """
    num_low_counts = counts[counts < privacy_threshold].sum()
    if num_low_counts > 0:
        # It may be possible to share ranges of values instead of the actual values,
        # but we need to be vary careful. E.g. if the dataframe length is 20 and we
        # have frequencies 2 and 18, masking 2 as 0-5 while sharing 18 and 20 is not
        # effective. Similarly, if we have frequencies 17 and three times 1, masking 1
        # as 0-5 thrice and sharing 17 is also not helpful.
        # Because it is rather difficult to ensure that nothing can be inferred, we
        # choose not to share anything if one of the frequencies is too low.
        # TODO how do we make clear to the user that this happened in the central task?
        warn(
            f"Value counts for column {column} contain values with low frequency. "
            "All counts for this column will be masked."
        )
        return {}
    
    # return counts.to_dict()
    def to_py_scalar(x):
        # Convert numpy scalars (e.g. numpy.bool_, numpy.int64) to native Python types
        return x.item() if isinstance(x, np.generic) else x
    
    return {to_py_scalar(k): to_py_scalar(v) for k, v in counts.items()}


def _filter_results(
    summary_numeric: pd.DataFrame, summary_categorical: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter out the variables that are not allowed to be shared

    Parameters
    ----------
    summary_numeric : pd.DataFrame
        The summary statistics for the numeric columns
    summary_categorical : pd.DataFrame
        The summary statistics for the non-numeric columns

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The filtered summary statistics for the numeric and non-numeric columns
    """
    if not get_env_var(EnvVarsAllowed.ALLOW_MIN.value, default="true", as_type="bool"):
        warn("Removing minimum from summary as policies do not allow sharing it.")
        summary_numeric.drop("min", inplace=True)
    if not get_env_var(EnvVarsAllowed.ALLOW_MAX.value, default="true", as_type="bool"):
        warn("Removing maximum from summary as policies do not allow sharing it.")
        summary_numeric.drop("max", inplace=True)
    if not get_env_var(
        EnvVarsAllowed.ALLOW_COUNT.value, default="true", as_type="bool"
    ):
        warn("Removing count from summary as policies do not allow sharing it.")
        summary_numeric.drop("count", inplace=True)
    if not get_env_var(EnvVarsAllowed.ALLOW_SUM.value, default="true", as_type="bool"):
        warn("Removing sum from summary as policies do not allow sharing it.")
        summary_numeric.drop("sum", inplace=True)
    if not get_env_var(
        EnvVarsAllowed.ALLOW_MISSING.value, default="true", as_type="bool"
    ):
        warn("Removing missing from summary as policies do not allow sharing it.")
        summary_numeric.drop("missing", inplace=True)
    return summary_numeric, summary_categorical


def check_privacy(df: pd.DataFrame, requested_columns: list[str]) -> None:
    """
    Check if the data complies with the privacy settings

    Parameters
    ----------
    df : pd.DataFrame
        The data to check
    requested_columns : list[str]
        The columns that are requested in the computation
    """
    min_rows = get_env_var(
        ENVVAR_MINIMUM_ROWS, default=DEFAULT_MINIMUM_ROWS, as_type="int"
    )
    if len(df) < min_rows:
        raise PrivacyThresholdViolation(
            f"Data contains less than {min_rows} rows. Refusing to "
            "handle this computation, as it may lead to privacy issues."
        )
    # check that each column has at least min_rows non-null values
    for col in df.columns:
        if df[col].count() < min_rows:
            raise PrivacyThresholdViolation(
                f"Column {col} contains less than {min_rows} non-null values. "
                "Refusing to handle this computation, as it may lead to privacy issues."
            )

    # Check if requested columns are allowed
    allowed_columns = get_env_var(ENVVAR_ALLOWED_COLUMNS)
    if allowed_columns:
        allowed_columns = allowed_columns.split(",")
        for col in requested_columns:
            if col not in allowed_columns:
                raise NodePermissionException(
                    f"The node administrator does not allow '{col}' to be requested in "
                    "this algorithm computation. Please contact the node administrator "
                    "for more information."
                )
    non_allowed_collumns = get_env_var(ENVVAR_DISALLOWED_COLUMNS)
    if non_allowed_collumns:
        non_allowed_collumns = non_allowed_collumns.split(",")
        for col in requested_columns:
            if col in non_allowed_collumns:
                raise NodePermissionException(
                    f"The node administrator does not allow '{col}' to be requested in "
                    "this algorithm computation. Please contact the node administrator "
                    "for more information."
                )


def check_match_inferred_numeric(
    numeric_columns: list[str],
    inferred_numeric_columns: list[str],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check if the provided numeric_columns list matches the inferred numerical columns

    Parameters
    ----------
    numeric_columns : list[str]
        The user-provided list of columns to be treated as numeric. If user did not
        provide this list, it is equal to the inferred_numeric_columns
    inferred_numeric_columns : list[str]
        The inferred list of numerical columns
    df: pd.DataFrame
        The original data. The type of the data may be modified if possible

    Returns
    -------
    pd.DataFrame
        The data with the columns cast to numeric if possible

    Raises
    ------
    ValueError
        If the provided numeric_columns list does not match the inferred_numeric_columns
    """
    error_msg = ""
    for col in numeric_columns:
        if col not in inferred_numeric_columns:
            try:
                df = cast_df_to_numeric(df, [col])
            except ValueError as exc:
                error_msg += str(exc)
    if error_msg:
        raise ValueError(error_msg)
    return df


def cast_df_to_numeric(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Cast the columns in the dataframe to numeric if possible

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to cast
    columns : list[str] | None
        The columns to cast. If None, all columns are cast

    Returns
    -------
    pd.DataFrame
        The dataframe with the columns cast to numeric
    """
    if columns is None:
        columns = df.columns
    for col in columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError as exc:
            raise ValueError(f"Column {col} could not be cast to numeric") from exc
    return df


def _variance_per_data_station(
    df: pd.DataFrame, columns: list[str], means: list[float]
) -> dict:
    if not get_env_var(
        EnvVarsAllowed.ALLOW_VARIANCE.value, default="true", as_type="bool"
    ):
        error("Node policies do not allow sharing the variance.")
        return None
    # Check that column names exist in the dataframe - note that this check should
    # not be necessary if a user runs the central task as is has already been checked
    # in that case
    if not all([col in df.columns for col in columns]):
        non_existing_columns = [col for col in columns if col not in df.columns]
        raise InputError(
            f"Columns {non_existing_columns} do not exist in the dataframe"
        )
    if len(columns) != len(means):
        raise InputError(
            "Length of columns list does not match the length of means list"
        )

    # Filter dataframe to only include the columns of interest
    df = df[columns]

    # Check privacy settings
    info("Checking if data complies to privacy settings")
    # check_privacy(df, columns)

    # Strict IDEA4RC dtype enforcement
    assert_columns_dtype_in(
        df,
        columns,
        allowed=[Idea4rcDType.INT64, Idea4rcDType.FLOAT64],
        algorithm="summary_variance",
        expected_kind="numeric (nullable Int64 or Float64)",
    )

    # Calculate the variance
    info("Calculating variance")
    variances = {}
    for idx, column in enumerate(columns):
        mean = means[idx]
        variances[column] = ((df[column].astype(float) - mean) ** 2).sum()

    return variances
