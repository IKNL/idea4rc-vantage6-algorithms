from typing import Any
from importlib import import_module

import pandas as pd
import numpy as np

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.decorator import algorithm_client, dataframes
from vantage6.algorithm.tools.exceptions import AlgorithmExecutionError, InputError
from vantage6.algorithm.client import AlgorithmClient

from .decorator import new_data_decorator


# Temporary disable some privacy settings that are defined in the v6-summary-py
_summary = import_module("v6-summary-py")
_summary.utils.DEFAULT_MINIMUM_ROWS = 0
_summary.utils.DEFAULT_PRIVACY_THRESHOLD = 0

_summary.partial_summary.DEFAULT_MINIMUM_ROWS = 0
_summary.partial_summary.DEFAULT_PRIVACY_THRESHOLD = 0


@algorithm_client
def summary(
    client: AlgorithmClient,
    columns: list[str] | None = None,
    is_numeric: list[bool] | None = None,
    organizations_to_include: list[int] | None = None,
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
    is_numeric : list[bool] | None
        Whether each of the columns is numeric or not. If not given, the algorithm will
        try to infer the type of the columns.
    organizations_to_include : list[int] | None
        The organizations to include in the task. If not given, all organizations
        in the collaboration are included.
    """
    if is_numeric and len(is_numeric) != len(columns):
        raise InputError(
            "Length of is_numeric list does not match the length of columns list"
        )

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
        "method": "summary_per_data_station",
        "kwargs": {
            "columns": columns,
            "is_numeric": is_numeric,
        },
    }

    # create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        input_=input_,
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
    cohort_names = results[0].keys()

    for cohort_name in cohort_names:
        cohort_results = [result[cohort_name] for result in results]
        all_cohort_results[cohort_name] = _aggregate_partial_summaries(cohort_results)

        numerical_columns = list(all_cohort_results[cohort_name]["numeric"].keys())
        # compute the variance now that we have the mean
        means[cohort_name] = [
            all_cohort_results[cohort_name]["numeric"][column]["mean"]
            for column in numerical_columns
        ]
        info(f"n num cols: {len(numerical_columns)}")
        info(f"n means: {len(means[cohort_name])}")

    info("debugger")
    info(numerical_columns)
    info(len(means))

    task = client.task.create(
        input_={
            "method": "variance_per_data_station",
            "kwargs": {
                "columns": numerical_columns,
                "means": means,
            },
        },
        organizations=organizations_to_include,
        name="Subtask variance",
        description="Compute variance per data station",
    )
    variance_results = client.wait_for_results(task_id=task.get("id"))

    # add the standard deviation to the results
    for cohort_name in cohort_names:
        cohort_variance_results = [result[cohort_name] for result in variance_results]
        all_cohort_results[cohort_name] = _add_sd_to_results(
            all_cohort_results[cohort_name], cohort_variance_results, numerical_columns
        )

    # return the final results of the algorithm
    return all_cohort_results


def _aggregate_partial_summaries(results: list[dict]) -> dict:
    """Aggregate the partial summaries of all nodes.

    Parameters
    ----------
    results : list[dict]
        The partial summaries of all nodes.
    """
    info("Aggregating partial summaries")
    aggregated_summary = {}
    is_first = True
    for result in results:
        if result is None:
            raise AlgorithmExecutionError(
                "At least one of the nodes returned invalid result. Please check the "
                "logs."
            )
        if is_first:
            # copy results. Only convert num complete rows per node to a list so that
            # we can add the other nodes to it later
            aggregated_summary = result
            aggregated_summary["num_complete_rows_per_node"] = [
                result["num_complete_rows_per_node"]
            ]
            for column in result["numeric"]:
                aggregated_summary["numeric"][column]["median"] = [
                    result["numeric"][column]["median"]
                ]
                aggregated_summary["numeric"][column]["q_25"] = [
                    result["numeric"][column]["q_25"]
                ]
                aggregated_summary["numeric"][column]["q_75"] = [
                    result["numeric"][column]["q_75"]
                ]
            is_first = False
            continue

        # aggregate data for numeric columns
        for column in result["numeric"]:
            aggregated_dict = aggregated_summary["numeric"][column]
            aggregated_dict["count"] += result["numeric"][column]["count"]
            aggregated_dict["min"] = min(
                aggregated_summary["numeric"][column]["min"],
                result["numeric"][column]["min"],
            )
            aggregated_dict["max"] = max(
                aggregated_summary["numeric"][column]["max"],
                result["numeric"][column]["max"],
            )
            aggregated_dict["missing"] += result["numeric"][column]["missing"]
            aggregated_dict["sum"] += result["numeric"][column]["sum"]
            aggregated_dict["median"].append(result["numeric"][column]["median"])
            aggregated_dict["q_25"].append(result["numeric"][column]["q_25"])
            aggregated_dict["q_75"].append(result["numeric"][column]["q_75"])

        # aggregate data for categorical columns
        for column in result["categorical"]:
            aggregated_dict = aggregated_summary["categorical"][column]
            aggregated_dict["count"] += result["categorical"][column]["count"]
            aggregated_dict["missing"] += result["categorical"][column]["missing"]

        # add the number of complete rows for this node
        aggregated_summary["num_complete_rows_per_node"].append(
            result["num_complete_rows_per_node"]
        )

        # add the unique values
        for column in result["counts_unique_values"]:
            if column not in aggregated_summary["counts_unique_values"]:
                aggregated_summary["counts_unique_values"][column] = {}
            for value, count in result["counts_unique_values"][column].items():
                if value not in aggregated_summary["counts_unique_values"][column]:
                    aggregated_summary["counts_unique_values"][column][value] = 0
                aggregated_summary["counts_unique_values"][column][value] += count

    # now that all data is aggregated, we can compute the mean
    for column in aggregated_summary["numeric"]:
        aggregated_dict = aggregated_summary["numeric"][column]
        if aggregated_dict["count"]:
            aggregated_dict["mean"] = aggregated_dict["sum"] / aggregated_dict["count"]
        else:
            aggregated_dict["mean"] = 0  # TODO this is terrible, we should not do this

    return aggregated_summary


def _add_sd_to_results(
    results: dict, variance_results: list[dict], numerical_columns: list[str]
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
            sum_variance += node_results[column]
        if results["numeric"][column]["count"] > 1:
            variance = sum_variance / (results["numeric"][column]["count"] - 1)
        else:
            variance = 0  # TODO THIS IS TERRIBLE
        results["numeric"][column]["std"] = variance**0.5
    return results


# Do not provide the columns as we want all columns to be included
@dataframes
def summary_per_data_station(
    dataframes: dict[str, pd.DataFrame], *args, **kwargs
) -> dict:
    dfs = dataframes.values()
    cohort_names = dataframes.keys()
    results = {}
    for df, name in zip(dfs, cohort_names):
        results[name] = _summary.partial_summary._summary_per_data_station(
            df, *args, **kwargs
        )
        # Add median and quantiles (0.25, 0.75)
        for var in results[name]["numeric"]:
            results[name]["numeric"][var]["median"] = float(np.nanmedian(df[var]))
            results[name]["numeric"][var]["q_25"] = float(np.nanquantile(df[var], 0.25))
            results[name]["numeric"][var]["q_75"] = float(np.nanquantile(df[var], 0.75))

    return results


@dataframes
def variance_per_data_station(
    dataframes: dict[str, pd.DataFrame],
    means: dict[list[float]],
    *args,
    **kwargs,
) -> dict:
    dfs = dataframes.values()
    cohort_names = dataframes.keys()
    results = {}
    info(kwargs)
    info(means)
    for df, name in zip(dfs, cohort_names):
        info("*" * 80)
        info(name)
        info(means[name])
        info("*" * 80)
        results[name] = _summary.partial_variance._variance_per_data_station(
            df, means=means[name], *args, **kwargs
        )
    return results
