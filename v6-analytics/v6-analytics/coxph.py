import math

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


@federated
@dataframes
@metadata
def get_unique_event_times(dataframes: dict[str, pd.DataFrame], metadata: RunMetaData, time_col: str, outcome_col: str) -> dict[str, pd.DataFrame]:
    results = {}
    for name, df in dataframes.items():
        results[name] = _get_unique_event_times(df, time_col, outcome_col, metadata)
    return results

@federated
@dataframes
def compute_summed_z(dataframes: dict[str, pd.DataFrame], outcome_col: str, expl_vars: list[str]) -> dict[str, dict]:
    results = {}
    for name, df in dataframes.items():
        results[name] = _compute_summed_z(df, outcome_col, expl_vars)
    return results


@federated
@dataframes
def perform_iteration(dataframes: dict[str, pd.DataFrame], time_col: str, expl_vars: list[str], beta: list[float], unique_time_events: list[float]) -> dict[str, dict]:
    results = {}
    for name, df in dataframes.items():
        results[name] = _perform_iteration(df, time_col, expl_vars, beta, unique_time_events)
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

    if df[outcome_col].notnull().sum() <= 10:
        warn("Sub-task was not executed because the number of samples is too small (n <= 10)")
        return {"N-Threshold not met": metadata.organization_id}

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
def central(
        client: AlgorithmClient, time_col, outcome_col, expl_vars, organization_ids):
    """
    This function is the central part of the algorithm. It performs the main computation and coordination tasks.

    Parameters:
    client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    time_col (str): The name of the column in the DataFrame that contains the time data.
    outcome_col (str): The name of the column in the DataFrame that contains the outcome data.
    expl_vars (list): A list of explanatory variables to be used in the computation.
    organization_ids (list): A list of organization IDs that participate in the collaboration.

    Returns:
    pandas.DataFrame: A DataFrame containing the results of the computation.
    """

    # Collect all organization that participate in this collaboration unless specified
    if not isinstance(organization_ids, list):
        organisations = client.organization.list()
        ids = [organisation.get("id") for organisation in organisations]
    else:
        ids = organization_ids

    # Create a list to store the IDs of organizations that do not meet privacy guards
    excluded_ids = []

    info(f'Sending task to organizations {ids}')

    n_covs = len(expl_vars)
    epochs = 10

    n_loops = 0
    n_threshold_met = False
    while not n_threshold_met:
        # This list represents the organizations that will be excluded in the following loop
        _excluded_ids = []
        if n_loops > 2:
            error("Sample size violations should be eliminated yet criteria are not met. Exiting")
            raise ValueError("Sample size violations should be eliminated yet criteria are not. Exiting")

        n_loops += 1
        # Create a subtask for all selected organizations in the collaboration.
        info("Creating subtask for all selected organizations in the collaboration")
        task = client.task.create(
            method="get_unique_event_times",
            arguments={
                "time_col": time_col,
                "outcome_col": outcome_col
            },
            organizations=ids,
            name="Unique event times",
            description="Getting unique event times and their counts"
        )

        # Wait for the node to return results of the subtask.
        info("Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        info("Results obtained!")

        unique_time_events = []
        for output in results:

            # Exclude organizations that do not meet the N-threshold
            if "N-Threshold not met" in output:
                warn(f"Insufficient samples for organization {output['N-Threshold not met']}. "
                     f"Excluding organization from analysis.")
                ids.remove(output["N-Threshold not met"])
                excluded_ids.append(output["N-Threshold not met"])
                _excluded_ids.append(output["N-Threshold not met"])
                continue

            output = pd.DataFrame.from_dict(output["times"])
            unique_time_events.append(output)

        if len(_excluded_ids) == 0:
            n_threshold_met = True
        elif len(ids) == 0:
            warn("No organizations meet the minimal sample size threshold, returning NaN.")
            return {"excluded_organizations": excluded_ids, "table": np.nan}

    aggregated_time_events = pd.concat(unique_time_events)
    aggregated_time_events = aggregated_time_events.groupby(time_col, as_index=False).sum()

    # Get the list of unique_time_events
    unique_time_events = aggregated_time_events[time_col].tolist()

    # create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        method="compute_summed_z",
        arguments={
            "outcome_col": outcome_col,
            "expl_vars": expl_vars,
        },
        organizations=ids,
        name="Summed Z statistic",
        description="Computing the summed Z statistic"
    )

    # wait for node to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")

    z_sum = 0
    for output in results:
        z_sum += pd.Series(output["sum"])

    beta = np.zeros(n_covs)

    for epoch in range(epochs):

        # JSON-serialize beta for Vantage6
        beta = beta.tolist()

        # De-serialise beta again
        beta = np.array(beta)

        # create a subtask for all organizations in the collaboration.
        info("Creating subtask for all organizations in the collaboration")
        task = client.task.create(
            method="perform_iteration",
            arguments={
                'time_col': time_col,
                "expl_vars": expl_vars,
                'beta': beta,
                'unique_time_events': unique_time_events
            },
            organizations=ids,
            name="Start iteration",
            description="Iterating to find the optimal beta"
        )

        # wait for node to return results of the subtask.
        info("Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        info("Results obtained!")

        summed_agg1 = 0
        summed_agg2 = 0
        summed_agg3 = 0

        for output in results:
            summed_agg1 += np.array(output['agg1'])
            summed_agg2 += np.array(pd.DataFrame.from_dict(output['agg2']))
            summed_agg3 += np.array([np.array(lst) for lst in output['agg3']])

        primary_derivative, secondary_derivative = compute_derivatives(summed_agg1, summed_agg2, summed_agg3,
                                                                       aggregated_time_events,
                                                                       z_sum)

        beta_old = np.array(beta)
        beta = beta_old - solve(secondary_derivative, primary_derivative)
        delta = float(max(abs(beta - beta_old)))

        if math.isnan(delta):
            info("Delta has turned into a NaN?")
            break

        if delta <= 0.000001:
            info("Betas have settled! Finished iterating!")
            break

    # Computing the standard errors
    SErrors = []
    fisher = np.linalg.inv(-secondary_derivative)
    for k in range(fisher.shape[0]):
        SErrors.append(np.sqrt(fisher[k, k]))

    # Calculating P and Z values
    zvalues = (np.exp(beta) - 1) / np.array(SErrors)
    pvalues = 2 * norm.cdf(-abs(zvalues))

    # Calculate overall model significance using Wald test
    # Reference: Andersen & Gill (1982) "Cox's regression model for counting processes"
    degrees_of_freedom = len(beta)
    wald_statistic = np.dot(beta, np.dot(-secondary_derivative, beta))
    overall_p_value = chi2.sf(wald_statistic, degrees_of_freedom)

    # Compute AIC for model comparison
    # Reference: Cox (1972) "Regression models and life tables" - defines partial likelihood
    try:
        # Cox partial log-likelihood: L(β) = Σ[β'x_i - log(Σ_j exp(β'x_j))]
        # First term: linear predictor contribution for all events
        linear_part = np.dot(z_sum, beta)

        # Second term: log of risk set sums (denominator terms)
        # final summed_agg1 contains the risk set denominators at the converged β values
        risk_set_part = 0
        if hasattr(summed_agg1, '__len__') and len(summed_agg1) > 0:
            for i in range(len(aggregated_time_events)):
                if i < len(summed_agg1) and summed_agg1[i] > 0:
                    freq = aggregated_time_events.iloc[i]['freq']
                    # Check for numerical issues before computing log
                    if summed_agg1[i] <= 0:
                        # Risk of negative or zero due to noise in DP setting
                        warn(f"Risk set sum is non-positive at time index {i}: {summed_agg1[i]}")
                        continue
                    risk_set_part += freq * np.log(summed_agg1[i])

        log_likelihood = linear_part - risk_set_part
        n_params = len(beta)  # degrees of freedom

        # Check for numerical issues in log-likelihood
        if np.isnan(log_likelihood) or np.isinf(log_likelihood):
            raise ValueError(f"Invalid log-likelihood: {log_likelihood}")

        # AIC = -2 * log-likelihood + 2 * k (Akaike, 1974)
        aic = -2 * log_likelihood + 2 * n_params

    except (ValueError, IndexError, FloatingPointError) as e:
        warn(f"Could not compute AIC due to numerical/data issue: {e}")
        aic = np.nan
        n_params = len(beta)
    except Exception as e:
        warn(f"Unexpected error computing AIC: {e}")
        aic = np.nan
        n_params = len(beta)

    # 95%CI = beta +- 1.96 * SE
    results = pd.DataFrame(
        np.array([np.around(beta, 5), np.around(np.exp(beta), 5), np.around(np.array(SErrors), 5)]).T,
        columns=["Coef", "Exp(coef)", "SE"])
    results['Var'] = expl_vars
    results["lower_CI"] = np.around(np.exp(results["Coef"] - 1.96 * results["SE"]), 5)
    results["upper_CI"] = np.around(np.exp(results["Coef"] + 1.96 * results["SE"]), 5)
    results["Z"] = zvalues
    results["p-value"] = pvalues
    results = results.set_index("Var")

    # Collect warnings for perfect prediction
    warnings = []
    threshold = 10
    for idx, row in results.iterrows():
        coef = row["Coef"]
        se = row["SE"]
        if (
                abs(coef) > threshold or np.isinf(coef) or np.isnan(coef) or
                abs(se) > threshold or np.isinf(se) or np.isnan(se)
        ):
            msg = (
                f"Warning: Covariate '{row['Var']}' may perfectly predict the event "
                f"(coef={coef}, SE={se}). Results may be unreliable."
            )
            warn(msg)
            warnings.append(msg)

    return {"included_organizations": ids,
            "excluded_organizations": excluded_ids,
            "model": results.to_json(),
            "overall_p_value": float(overall_p_value),
            "aic": float(aic),
            "degrees_of_freedom": int(n_params),
            "warnings": warnings}


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