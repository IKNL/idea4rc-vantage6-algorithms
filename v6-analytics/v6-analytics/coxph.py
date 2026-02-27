import numpy as np
import pandas as pd

from scipy.stats import norm, chi2
from scipy.linalg import solve

from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.decorator.metadata import RunMetaData
from vantage6.algorithm.decorator import (
    federated,
    dataframes,
    algorithm_client,
    central,
    metadata
)


def _filter_dataframes_on_names(
    dataframes: dict[str, pd.DataFrame], use_dataframe_names: list[str] | None
) -> dict[str, pd.DataFrame]:
    """Filter dataframes to only those in use_dataframe_names, preserving order."""
    if not use_dataframe_names:
        return dataframes
    return {name: dataframes[name] for name in use_dataframe_names if name in dataframes}


@federated
@metadata
@dataframes
def get_unique_event_times(
    dataframes: dict[str, pd.DataFrame],
    metadata: RunMetaData,
    time_col: str,
    outcome_col: str,
    use_dataframe_names: list[str] | None = None,
) -> dict[str, dict]:
    dataframes = _filter_dataframes_on_names(dataframes, use_dataframe_names)
    results = {}
    for name, df in dataframes.items():
        results[name] = _get_unique_event_times(df, time_col, outcome_col, metadata)
    return results


@federated
@dataframes
def compute_summed_z(
    dataframes: dict[str, pd.DataFrame],
    outcome_col: str,
    expl_vars: list[str],
    use_dataframe_names: list[str] | None = None,
) -> dict[str, dict]:
    dataframes = _filter_dataframes_on_names(dataframes, use_dataframe_names)
    results = {}
    for name, df in dataframes.items():
        results[name] = _compute_summed_z(df, outcome_col, expl_vars)
    return results


@federated
@dataframes
def perform_iteration(
    dataframes: dict[str, pd.DataFrame],
    time_col: str,
    expl_vars: list[str],
    beta: dict[str, list[float]],
    unique_time_events: dict[str, list[float]],
    use_dataframe_names: list[str] | None = None,
) -> dict[str, dict]:
    dataframes = _filter_dataframes_on_names(dataframes, use_dataframe_names)
    results = {}
    for name, df in dataframes.items():
        results[name] = _perform_iteration(
            df, time_col, expl_vars, beta[name], unique_time_events[name]
        )
    return results


def _get_unique_event_times(df: pd.DataFrame, time_col: str, outcome_col: str, metadata: RunMetaData) -> dict:
    """
    This function retrieves unique event times from the provided DataFrame.
    If the number of samples is too small, the sub-task is halted and returns the organization ID.

    Parameters:
    client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    df (pandas.DataFrame): The DataFrame containing the data.
    time_col (str): The name of the column in the DataFrame that contains the time data.
    outcome_col (str): The name of the column in the DataFrame that contains the outcome data.

    Returns:
    dict: A dictionary containing a DataFrame of unique event times,
    or a message indicating that the subtask was not executed for privacy reasons.
    """
    info("Computing unique event times")

    # if df[outcome_col].notnull().sum() <= 10:
    #     warn("Sub-task was not executed because the number of samples is too small (n <= 10)")
    #     return {"N-Threshold not met": metadata.organization_id}

    times = df[df[outcome_col] == 1].groupby(time_col, as_index=False).count()
    times = times.sort_values(by=time_col)[[time_col, outcome_col]]
    times['freq'] = times[outcome_col]
    times = times.drop(columns=outcome_col)
    return {'times': times.to_dict()}


def _compute_summed_z(df: pd.DataFrame, outcome_col, expl_vars):
    """
    This function computes the sum of the specified explanatory variables for the outcome events.

    Parameters:
    client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    df (pandas.DataFrame): The DataFrame containing the data.
    outcome_col (str): The name of the column in the DataFrame that contains the outcome data.
    expl_vars (list): A list of explanatory variables to be used in the computation.

    Returns:
    dict: A dictionary containing the sum of the explanatory variables for the outcome events.
    """
    info("Computing summed z statistics")
    z_sum = (df[df[outcome_col] == 1][expl_vars].sum().to_dict())
    return {'sum': z_sum}


def _perform_iteration(df: pd.DataFrame, time_col, expl_vars, beta, unique_time_events):
    """
    This function performs an iteration of the algorithm, computing the necessary aggregates.

    Parameters:
    client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    df (pandas.DataFrame): The DataFrame containing the data.
    time_col (str): The name of the column in the DataFrame that contains the time data.
    expl_vars (list): A list of explanatory variables to be used in the computation.
    beta (numpy.ndarray): The current estimate of the beta coefficients.
    unique_time_events (list): A list of unique time events.

    Returns:
    dict: A dictionary containing the aggregates computed during the iteration.
    """
    info("Computing aggregates for the derivation of the partial likelihood")
    # Deserialize beta values
    beta = np.array(beta)
    num_unique_time_events = len(unique_time_events)
    num_explanatory_vars = len(expl_vars)

    agg1 = []
    agg2 = []
    agg3 = []

    for i in range(num_unique_time_events):
        R_i = df[df[time_col] >= unique_time_events[i]][expl_vars]
        # Check if R_i is empty
        if not R_i.empty:
            ebz = np.exp(np.dot(np.array(R_i), beta))
            agg1.append(sum(ebz))
            func = lambda x: np.asarray(x) * np.asarray(ebz)
            z_ebz = R_i.apply(func)
            agg2.append(z_ebz.sum())

            summed = np.zeros((num_explanatory_vars, num_explanatory_vars))
            for j in range(len(R_i)):
                summed = summed + np.outer(np.array(z_ebz)[j], np.array(R_i)[j].T)
            agg3.append(summed)

        else:
            agg1.append(0)
            agg2.append(pd.Series(np.zeros(num_explanatory_vars), index=expl_vars))
            agg3.append(np.zeros((num_explanatory_vars, num_explanatory_vars)))

    # JSON-serialize the results
    agg2 = pd.DataFrame(agg2).to_dict()
    agg3 = [array.tolist() for array in agg3]

    return {'agg1': agg1,
            'agg2': agg2,
            'agg3': agg3}

#
# Central stuff
#
@central
@algorithm_client
def coxph_central(
    client: AlgorithmClient, time_col, outcome_col, expl_vars, organization_ids
):
    """
    This function is the central part of the algorithm. It performs the main
    computation and coordination tasks for one or more dataframes (cohorts).
    Results are unpacked per dataframe and dataframes that have converged are
    removed from the iteration loop.

    Returns:
    dict: {"cohorts": {df_name: {...}}, "details": {"iterations": ..., "all_converged": ...}}
    """

    if not isinstance(organization_ids, list):
        organisations = client.organization.list()
        ids = [organisation.get("id") for organisation in organisations]
    else:
        ids = list(organization_ids)

    info(f"Sending task to organizations {ids}")

    n_covs = len(expl_vars)
    epochs = 25
    tolerance = 1e-6

    # --- get_unique_event_times: unpack per dataframe ---
    n_loops = 0
    while True:
        if n_loops > 2:
            error(
                "Sample size violations should be eliminated yet criteria are not met. Exiting"
            )
            raise ValueError(
                "Sample size violations should be eliminated yet criteria are not. Exiting"
            )
        n_loops += 1

        task = client.task.create(
            method="get_unique_event_times",
            arguments={"time_col": time_col, "outcome_col": outcome_col},
            organizations=ids,
            name="Unique event times",
            description="Getting unique event times and their counts",
        )
        info("Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        info("Results obtained!")

        if not results:
            warn("No results returned from get_unique_event_times.")
            return {"cohorts": {}, "details": {"iterations": 0, "all_converged": False}}

        dataframe_names = list(results[0].keys())
        excluded_ids = {df_name: [] for df_name in dataframe_names}
        unique_time_events_raw = {df_name: [] for df_name in dataframe_names}

        for org_id, org_result in zip(ids, results):
            for df_name, output in org_result.items():
                if df_name not in excluded_ids:
                    excluded_ids[df_name] = []
                if df_name not in unique_time_events_raw:
                    unique_time_events_raw[df_name] = []
                if "N-Threshold not met" in output:
                    warn(
                        f"Insufficient samples for organization {org_id} (dataframe {df_name}). "
                        f"Excluding from this dataframe."
                    )
                    excluded_ids[df_name].append(org_id)
                elif "times" in output:
                    unique_time_events_raw[df_name].append(
                        pd.DataFrame.from_dict(output["times"])
                    )

        ids_included = {
            df_name: [i for i in ids if i not in excluded_ids[df_name]]
            for df_name in dataframe_names
        }
        any_excluded_this_round = any(
            excluded_ids[df_name] for df_name in dataframe_names
        )
        if not any_excluded_this_round:
            break
        if all(not ids_included[df_name] for df_name in dataframe_names):
            warn(
                "No organizations meet the minimal sample size threshold for any dataframe, returning NaN."
            )
            return {
                "cohorts": {},
                "details": {"iterations": 0, "all_converged": False},
                "excluded_organizations": excluded_ids,
            }

    aggregated_time_events = {}
    unique_time_events = {}
    for df_name in dataframe_names:
        if not unique_time_events_raw[df_name]:
            continue
        agg = pd.concat(unique_time_events_raw[df_name]).groupby(
            time_col, as_index=False
        ).sum()
        aggregated_time_events[df_name] = agg
        unique_time_events[df_name] = agg[time_col].tolist()

    active_dataframe_names = [
        df_name
        for df_name in dataframe_names
        if df_name in aggregated_time_events and ids_included[df_name]
    ]
    if not active_dataframe_names:
        return {
            "cohorts": {},
            "details": {"iterations": 0, "all_converged": False},
        }

    # --- compute_summed_z: unpack per dataframe ---
    task = client.task.create(
        method="compute_summed_z",
        arguments={
            "outcome_col": outcome_col,
            "expl_vars": expl_vars,
            "use_dataframe_names": active_dataframe_names,
        },
        organizations=ids,
        name="Summed Z statistic",
        description="Computing the summed Z statistic",
    )
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")

    z_sum = {}
    for df_name in active_dataframe_names:
        included_set = set(ids_included[df_name])
        total = None
        for org_id, org_result in zip(ids, results):
            if org_id not in included_set or df_name not in org_result:
                continue
            out = org_result[df_name]
            if "sum" not in out:
                continue
            s = pd.Series(out["sum"])
            total = s if total is None else total + s
        z_sum[df_name] = (
            total
            if total is not None
            else pd.Series(0.0, index=expl_vars)
        )

    betas = {df_name: np.zeros(n_covs) for df_name in active_dataframe_names}
    converged_results = {}
    iteration = 0

    # --- Iteration loop with convergence kick-out ---
    for epoch in range(epochs):
        if not active_dataframe_names:
            break
        iteration = epoch + 1
        info(f"Iteration {iteration}")

        task = client.task.create(
            method="perform_iteration",
            arguments={
                "time_col": time_col,
                "expl_vars": expl_vars,
                "beta": {
                    df_name: betas[df_name].tolist()
                    for df_name in active_dataframe_names
                },
                "unique_time_events": {
                    df_name: unique_time_events[df_name]
                    for df_name in active_dataframe_names
                },
                "use_dataframe_names": active_dataframe_names,
            },
            organizations=ids,
            name="Start iteration",
            description="Iterating to find the optimal beta",
        )
        info("Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        info("Results obtained!")

        summed_agg1 = {}
        summed_agg2 = {}
        summed_agg3 = {}
        for df_name in active_dataframe_names:
            info(f"Computing summed aggregates for dataframe {df_name}")
            included_set = set(ids_included[df_name])
            parts_agg1 = []
            parts_agg2 = []
            parts_agg3 = []
            for org_id, org_result in zip(ids, results):
                if org_id not in included_set or df_name not in org_result:
                    continue
                out = org_result[df_name]
                if "agg1" in out and "agg2" in out and "agg3" in out:
                    parts_agg1.append(np.array(out["agg1"]))
                    parts_agg2.append(np.array(pd.DataFrame.from_dict(out["agg2"])))
                    parts_agg3.append(
                        np.array([np.array(lst) for lst in out["agg3"]])
                    )
            if parts_agg1:
                summed_agg1[df_name] = sum(parts_agg1)
                summed_agg2[df_name] = sum(parts_agg2)
                summed_agg3[df_name] = sum(parts_agg3)
            else:
                summed_agg1[df_name] = np.array([])
                summed_agg2[df_name] = np.array([])
                summed_agg3[df_name] = np.array([])

        to_remove = []
        for df_name in active_dataframe_names:
            if df_name not in summed_agg1 or len(summed_agg1[df_name]) == 0:
                continue
            sag1 = summed_agg1[df_name]
            sag2 = summed_agg2[df_name]
            sag3 = summed_agg3[df_name]
            agg_te = aggregated_time_events[df_name]
            zs = z_sum[df_name]

            primary_derivative, secondary_derivative = compute_derivatives(
                sag1, sag2, sag3, agg_te, zs
            )
            beta_old = np.array(betas[df_name])
            try:
                beta_new = beta_old - solve(secondary_derivative, primary_derivative)
            except Exception:
                info(f"Solve failed for dataframe {df_name}, keeping current beta.")
                continue
            delta = float(np.max(np.abs(beta_new - beta_old)))

            if np.isnan(delta):
                info(f"Delta is NaN for dataframe {df_name}.")
                to_remove.append(df_name)
                converged_results[df_name] = _prepare_cohort_result(
                    df_name,
                    beta_old,
                    agg_te,
                    zs,
                    sag1,
                    secondary_derivative,
                    expl_vars,
                    ids_included[df_name],
                    excluded_ids[df_name],
                    converged=False,
                )
                continue

            info(f"Delta: {delta}")

            if delta <= tolerance:
                info(f"Betas have settled for dataframe {df_name}!")
                to_remove.append(df_name)
                converged_results[df_name] = _prepare_cohort_result(
                    df_name,
                    beta_new,
                    agg_te,
                    zs,
                    sag1,
                    secondary_derivative,
                    expl_vars,
                    ids_included[df_name],
                    excluded_ids[df_name],
                    converged=True,
                )
            else:
                betas[df_name] = beta_new

        for df_name in to_remove:
            active_dataframe_names.remove(df_name)

    all_converged = len(active_dataframe_names) == 0

    # Store any remaining (non-converged) at max iterations
    for df_name in active_dataframe_names[:]:
        if df_name not in converged_results and df_name in summed_agg1:
            sag1 = summed_agg1[df_name]
            if len(sag1) > 0:
                sag2 = summed_agg2[df_name]
                sag3 = summed_agg3[df_name]
                agg_te = aggregated_time_events[df_name]
                zs = z_sum[df_name]
                primary_derivative, secondary_derivative = compute_derivatives(
                    sag1, sag2, sag3, agg_te, zs
                )
                converged_results[df_name] = _prepare_cohort_result(
                    df_name,
                    betas[df_name],
                    agg_te,
                    zs,
                    sag1,
                    secondary_derivative,
                    expl_vars,
                    ids_included[df_name],
                    excluded_ids[df_name],
                    converged=False,
                )
        active_dataframe_names.remove(df_name)

    return {
        "cohorts": converged_results,
        "details": {
            "iterations": iteration,
            "all_converged": all_converged,
        },
    }


def compute_derivatives(summed_agg1, summed_agg2, summed_agg3, aggregated_time_events, z_sum):
    """
    This function computes the primary and secondary derivatives needed for the central algorithm.

    Parameters:
    summed_agg1 (numpy.ndarray): The aggregated sum of the first set of values.
    summed_agg2 (numpy.ndarray): The aggregated sum of the second set of values.
    summed_agg3 (numpy.ndarray): The aggregated sum of the third set of values.
    aggregated_time_events (pandas.DataFrame): The DataFrame containing the frequency of unique event times.
    z_sum (float): The summed Z statistic.

    Returns:
    tuple: A tuple containing the primary and secondary derivatives.

    """

    tot_p1 = 0
    tot_p2 = 0

    # Iterate over each row in the DataFrame
    for index, row in aggregated_time_events.iterrows():
        # Compute the primary derivative component
        s1 = row['freq'] * (summed_agg2[index] / summed_agg1[index])

        # Compute the first part of the secondary derivative component
        first_part = (summed_agg3[index] / summed_agg1[index])

        # Compute the second part of the secondary derivative component
        # The numerator is the outer product of agg2
        numerator = np.outer(summed_agg2[index], summed_agg2[index])
        denominator = summed_agg1[index] * summed_agg1[index]
        second_part = numerator / denominator

        s2 = row['freq'] * (first_part - second_part)

        tot_p1 += s1
        tot_p2 += s2

    # Compute the primary and secondary derivatives
    primary_derivative = z_sum - tot_p1
    secondary_derivative = -tot_p2

    return primary_derivative, secondary_derivative


def _prepare_cohort_result(
    df_name: str,
    beta: np.ndarray,
    aggregated_time_events: pd.DataFrame,
    z_sum: pd.Series,
    summed_agg1: np.ndarray,
    secondary_derivative: np.ndarray,
    expl_vars: list[str],
    ids_included: list,
    excluded_ids: list,
    converged: bool = True,
) -> dict:
    """Build the per-cohort result dict (model table, AIC, warnings, etc.)."""
    n_params = len(beta)
    SErrors = []
    fisher = np.linalg.inv(-secondary_derivative)
    for k in range(fisher.shape[0]):
        SErrors.append(np.sqrt(fisher[k, k]))

    zvalues = (np.exp(beta) - 1) / np.array(SErrors)
    pvalues = 2 * norm.cdf(-abs(zvalues))
    degrees_of_freedom = n_params
    wald_statistic = np.dot(beta, np.dot(-secondary_derivative, beta))
    overall_p_value = float(chi2.sf(wald_statistic, degrees_of_freedom))

    try:
        linear_part = np.dot(z_sum, beta)
        risk_set_part = 0
        if hasattr(summed_agg1, "__len__") and len(summed_agg1) > 0:
            for i in range(len(aggregated_time_events)):
                if i < len(summed_agg1) and summed_agg1[i] > 0:
                    freq = aggregated_time_events.iloc[i]["freq"]
                    if summed_agg1[i] <= 0:
                        warn(
                            f"Risk set sum is non-positive at time index {i}: {summed_agg1[i]}"
                        )
                        continue
                    risk_set_part += freq * np.log(summed_agg1[i])
        log_likelihood = linear_part - risk_set_part
        if np.isnan(log_likelihood) or np.isinf(log_likelihood):
            raise ValueError(f"Invalid log-likelihood: {log_likelihood}")
        aic = float(-2 * log_likelihood + 2 * n_params)
    except (ValueError, IndexError, FloatingPointError) as e:
        warn(f"Could not compute AIC due to numerical/data issue: {e}")
        aic = np.nan
    except Exception as e:
        warn(f"Unexpected error computing AIC: {e}")
        aic = np.nan

    results_df = pd.DataFrame(
        np.array(
            [
                np.around(beta, 5),
                np.around(np.exp(beta), 5),
                np.around(np.array(SErrors), 5),
            ]
        ).T,
        columns=["Coef", "Exp(coef)", "SE"],
    )
    results_df["Var"] = expl_vars
    results_df["lower_CI"] = np.around(
        np.exp(results_df["Coef"] - 1.96 * results_df["SE"]), 5
    )
    results_df["upper_CI"] = np.around(
        np.exp(results_df["Coef"] + 1.96 * results_df["SE"]), 5
    )
    results_df["Z"] = zvalues
    results_df["p-value"] = pvalues
    results_df = results_df.set_index("Var")

    warnings_list = []
    threshold = 10
    for idx, row in results_df.iterrows():
        coef, se = row["Coef"], row["SE"]
        if (
            abs(coef) > threshold
            or np.isinf(coef)
            or np.isnan(coef)
            or abs(se) > threshold
            or np.isinf(se)
            or np.isnan(se)
        ):
            msg = (
                f"Warning: Covariate '{idx}' may perfectly predict the event "
                f"(coef={coef}, SE={se}). Results may be unreliable."
            )
            warn(msg)
            warnings_list.append(msg)

    return {
        "included_organizations": ids_included,
        "excluded_organizations": excluded_ids,
        "model": results_df.to_json(),
        "overall_p_value": overall_p_value,
        "aic": aic,
        "degrees_of_freedom": int(n_params),
        "warnings": warnings_list,
        "converged": converged,
    }