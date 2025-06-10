import re

from enum import Enum
from typing import Any
from functools import reduce

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.genmod.families as families
import scipy.stats as stats

from formulaic import Formula

from vantage6.algorithm.tools.exceptions import (
    UserInputError,
    PrivacyThresholdViolation,
    NodePermissionException,
    AlgorithmExecutionError,
)

from vantage6.algorithm.tools.util import info, get_env_var
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.util import info, warn, get_env_var
from vantage6.algorithm.client import AlgorithmClient

from .decorator import new_data_decorator


# Constants for main function arguments
DEFAULT_MAX_ITERATIONS = 25
DEFAULT_TOLERANCE = 1e-8

# names of environment variables that can be defined to override default values
## minimum number of rows in the dataframe
ENVVAR_MINIMUM_ROWS = "GLM_MINIMUM_ROWS"
## whitelist of columns allowed to be requested
ENVVAR_ALLOWED_COLUMNS = "GLM_ALLOWED_COLUMNS"
## blacklist of columns not allowed to be requested
ENVVAR_DISALLOWED_COLUMNS = "GLM_DISALLOWED_COLUMNS"
### minimum number of organizations to include in the analysis
ENVVAR_MINIMUM_ORGANIZATIONS = "GLM_MINIMUM_ORGANIZATIONS"
### maximum percentage of number of variables relative to number of observations
### allowed in the model. If the number of variables exceeds this percentage,
### the model will not be run due to risks of data leakage through overfitting.
ENVVAR_MAX_PCT_PARAMS_OVER_OBS = "GLM_MAX_PCT_VARS_VS_OBS"

# default values for environment variables
DEFAULT_MINIMUM_ROWS = 3
DEFAULT_MINIMUM_ORGANIZATIONS = 1
DEFAULT_MAX_PCT_PARAMS_VS_OBS = 100


def _temp_fix_to_convert_vars_to_int(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    This is a temporary fix to convert the variables to int.
    """
    for df in dfs:
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            df[f"SURVIVAL_{i}YR"] = df[f"SURVIVAL_{i}YR"].astype(int)
            df[f"DEATH_{i}YR"] = df[f"DEATH_{i}YR"].astype(int)

    return dfs


@new_data_decorator
def compute_local_betas(
    dfs: list[pd.DataFrame],
    cohort_names: list[str],
    use_cohort_names: list[str],
    formula: str,
    family: str,
    is_first_iteration: bool,
    beta_coefficients: list[dict[str, float]] | None = None,
    categorical_predictors: list[str] | None = None,
    survival_sensor_column: str = None,
) -> dict:
    """
    Compute the local betas for the cohorts in use_cohort_names.
    """
    local_betas = {}

    # filter dfs and cohort_names to only include the ones in use_cohort_names
    if use_cohort_names:
        dfs, cohort_names = _filter_df_on_cohort_names(
            dfs, cohort_names, use_cohort_names
        )

    dfs = _temp_fix_to_convert_vars_to_int(dfs)

    # in the first iteration, beta_coefficients is None
    if not beta_coefficients:
        beta_coefficients = {cohort_name: None for cohort_name in cohort_names}

    for df, cohort_name in zip(dfs, cohort_names):
        info(f"cohort_name: {cohort_name}")
        info(f"betas_for_cohort: {beta_coefficients[cohort_name]}")
        info(f"formula: {formula}")
        info(f"family: {family}")
        info(f"is_first_iteration: {is_first_iteration}")
        info(f"categorical_predictors: {categorical_predictors}")
        info(f"survival_sensor_column: {survival_sensor_column}")
        local_betas[cohort_name] = _compute_local_betas(
            df,
            formula,
            family,
            is_first_iteration,
            beta_coefficients[cohort_name],
            categorical_predictors,
            survival_sensor_column,
        )
    return local_betas


@new_data_decorator
def compute_local_deviance(
    dfs: list[pd.DataFrame],
    cohort_names: list[str],
    use_cohort_names: list[str],
    formula: str,
    family: str,
    is_first_iteration: bool,
    global_average_outcome_var: list[float],
    beta_coefficients: list[dict[str, float]],
    beta_coefficients_previous: list[dict[str, float]] | None = None,
    categorical_predictors: list[str] | None = None,
    survival_sensor_column: str | None = None,
) -> dict:
    # filter dfs and cohort_names to only include the ones in use_cohort_names
    if use_cohort_names:
        dfs, cohort_names = _filter_df_on_cohort_names(
            dfs, cohort_names, use_cohort_names
        )

    dfs = _temp_fix_to_convert_vars_to_int(dfs)

    # in the first iteration, beta_coefficients_previous is None
    if not beta_coefficients_previous:
        beta_coefficients_previous = {cohort_name: None for cohort_name in cohort_names}

    local_deviance = {}
    for (
        df,
        cohort_name,
        betas_for_cohort,
        betas_previous_for_cohort,
        global_average_outcome_var_for_cohort,
    ) in zip(
        dfs,
        cohort_names,
        beta_coefficients.values(),
        beta_coefficients_previous.values(),
        global_average_outcome_var.values(),
    ):
        print(f"betas_for_cohort: {betas_for_cohort}")
        print(f"betas_previous_for_cohort: {betas_previous_for_cohort}")

        local_deviance[cohort_name] = _compute_local_deviance(
            df,
            formula,
            family,
            is_first_iteration,
            global_average_outcome_var_for_cohort,
            betas_for_cohort,
            betas_previous_for_cohort,
            categorical_predictors,
            survival_sensor_column,
        )

    return local_deviance


@algorithm_client
def glm(
    client: AlgorithmClient,
    family: str,
    outcome_variable: str | None = None,
    predictor_variables: list[str] | None = None,
    formula: str | None = None,
    categorical_predictors: list[str] = None,
    category_reference_values: dict[str, str] = None,
    survival_sensor_column: str = None,
    tolerance_level: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    organizations_to_include: list[int] = None,
) -> dict:
    """
    Central part of the GLM algorithm

    This function creates subtasks for all the organizations involved in the GLM
    computation to compute partial results on their data and aggregates these, over
    multiple cycles, until the GLM is converged.

    Parameters
    ----------
    client : AlgorithmClient
        The client object to interact with the server
    family : str, optional
        The exponential family to use for computing the GLM. The available families are
        Gaussian, Poisson, Binomial, and Survival.
    outcome_variable : str, optional
        The name of the outcome variable column, by default None. If not provided, the
        formula must be provided.
    predictor_variables : list[str], optional
        The names of the predictor variable columns, by default None. If not provided,
        the formula must be provided.
    formula : str, optional
        The formula to use for the GLM, by default None. If not provided, the
        outcome_variable and predictor_variables must be provided.
    categorical_predictors : list[str], optional
        The column names of the predictor variables that are categorical. All columns
        with string values are considered categorical by default - this option should
        be used for columns with numerical values that should be treated as categorical.
    category_reference_values : dict[str, str], optional
        The reference values for the categorical variables, by default None. If, for
        instance, the predictor variable 'A' is a categorical variable with values
        'a', 'b', and 'c', and we want 'a' to be the reference value, this dictionary
        should be {'A': 'a'}.
    survival_sensor_column : str, optional
        The survival_sensor_column value, by default None. Required if the family is
        'survival'.
    tolerance_level : float, optional
        The tolerance level for the convergence of the algorithm, by default 1e-8.
    max_iterations : int, optional
        The maximum number of iterations for the algorithm, by default 25.
    organizations_to_include : list[int], optional
        The organizations to include in the computation, by default None. If not
        provided, all organizations in the collaboration are included.

    Returns
    -------
    dict
        The results of the GLM computation, including the coefficients and details of
        the computation.
    """
    # select organizations to include
    if not organizations_to_include:
        organizations = client.organization.list()
        organizations_to_include = [
            organization.get("id") for organization in organizations
        ]

    _check_input(
        organizations_to_include,
        family,
        formula,
        outcome_variable,
        predictor_variables,
        survival_sensor_column,
    )

    if not formula and outcome_variable:
        formula = get_formula(
            outcome_variable,
            predictor_variables,
            category_reference_values,
            categorical_predictors,
        )

    # Initialize tracking of converged cohorts
    all_cohort_names = None  # Will be populated after first iteration
    active_cohort_names = []  # Will track non-converged cohorts
    converged_results = {}  # Will store results of converged cohorts

    # Iterate to find the coefficients
    iteration = 1
    # betas = {cohort_name: None for cohort_name in cohort_names}
    betas = None
    while iteration <= max_iterations:
        converged, new_betas, deviance, cohort_names = _do_iteration(
            iteration=iteration,
            client=client,
            formula=formula,
            family=family.lower(),
            categorical_predictors=categorical_predictors,
            survival_sensor_column=survival_sensor_column,
            tolerance_level=tolerance_level,
            organizations_to_include=organizations_to_include,
            betas_old=betas,
            use_cohort_names=active_cohort_names,
        )

        # On first iteration, initialize cohort tracking
        if iteration == 1:
            all_cohort_names = cohort_names
            active_cohort_names = cohort_names

        # Update betas and track converged cohorts
        betas = {
            cohort_name: new_betas[cohort_name]["beta_estimates"]
            for cohort_name in cohort_names
        }

        # Check convergence for each cohort
        for cohort in active_cohort_names[:]:  # Iterate over copy to allow removal
            if deviance[cohort]["new"] == 0 or (
                abs(deviance[cohort]["old"] - deviance[cohort]["new"])
                / deviance[cohort]["new"]
                < tolerance_level
            ):
                # Store results for converged cohort
                converged_results[cohort] = _prepare_cohort_results(
                    new_betas[cohort],
                    deviance[cohort],
                )
                active_cohort_names.remove(cohort)

        # terminate if all cohorts have converged or reached max iterations
        if not active_cohort_names:
            info(" - All cohorts converged!")
            break
        if iteration == max_iterations:
            warn(" - Maximum number of iterations reached!")
            # Store results for non-converged cohorts
            for cohort in active_cohort_names:
                converged_results[cohort] = _prepare_cohort_results(
                    new_betas[cohort],
                    deviance[cohort],
                    converged=False,
                )
            break
        iteration += 1

    return {
        "cohorts": converged_results,
        "details": {
            "iterations": iteration,
            "all_converged": len(active_cohort_names) == 0,
        },
    }


def _prepare_cohort_results(
    betas: dict, deviance: dict, converged: bool = True
) -> dict:
    """Prepare the final results for a single cohort."""
    betas_series = pd.Series(betas["beta_estimates"])
    std_errors = pd.Series(betas["std_error_betas"])
    zvalue = betas_series / std_errors

    if betas["is_dispersion_estimated"]:
        pvalue = 2 * stats.t.cdf(
            -np.abs(zvalue), betas["num_observations"] - betas["num_variables"]
        )
    else:
        pvalue = 2 * stats.norm.cdf(-np.abs(zvalue))

    # add back indices to pvalue
    pvalue = pd.Series(pvalue, index=betas_series.index)

    # create dataframe with results
    results = pd.DataFrame(
        {
            "beta": betas_series,
            "std_error": std_errors,
            "z_value": zvalue,
            "p_value": pvalue,
        }
    )

    return {
        "coefficients": results.to_dict(),
        "details": {
            "converged": converged,
            "dispersion": betas["dispersion"],
            "is_dispersion_estimated": betas["is_dispersion_estimated"],
            "deviance": deviance["new"],
            "null_deviance": deviance["null"],
            "num_observations": betas["num_observations"],
            "num_variables": betas["num_variables"],
        },
    }


def _do_iteration(
    iteration: int,
    client: AlgorithmClient,
    formula: str,
    family: str,
    categorical_predictors: list[str],
    survival_sensor_column: str,
    tolerance_level: int,
    organizations_to_include: list[int],
    use_cohort_names: list[str],
    betas_old: dict | None = None,
) -> tuple[bool, dict, dict, list[str]]:
    """
    Execute one iteration of the GLM algorithm for multiple cohorts

    Returns
    -------
    tuple[bool, dict, dict, list[str]]
        A tuple containing:
        - boolean indicating if all cohorts have converged
        - dictionary containing the new beta coefficients per cohort
        - dictionary containing the deviance per cohort
        - list of all cohort names
    """
    # print iteration header to logs
    _log_header(iteration)

    # compute beta coefficients
    partial_betas = _compute_local_betas_task(
        client,
        formula,
        family,
        categorical_predictors,
        survival_sensor_column,
        iter_num=iteration,
        organizations_to_include=organizations_to_include,
        betas=betas_old,
        use_cohort_names=use_cohort_names,
    )
    info(" - Partial betas obtained!")

    # Get all cohort names on first iteration
    cohort_names = list(partial_betas[0].keys()) if iteration == 1 else use_cohort_names

    # compute central betas from the partial betas
    info("Computing central betas")
    new_betas = {}
    for cohort in cohort_names:
        info(f"  cohort: {cohort}")
        cohort_partials = [result[cohort] for result in partial_betas]
        new_betas[cohort] = _compute_central_betas(cohort_partials, family)
    info(" - Central betas obtained!")

    # compute the deviance for each cohort
    info("Computing deviance")
    deviance_partials = _compute_partial_deviance(
        client=client,
        formula=formula,
        family=family,
        categorical_predictors=categorical_predictors,
        iter_num=iteration,
        survival_sensor_column=survival_sensor_column,
        beta_estimates={
            cohort: betas["beta_estimates"] for cohort, betas in new_betas.items()
        },
        beta_estimates_previous=betas_old,
        global_average_outcome_var={
            cohort: betas["y_average"] for cohort, betas in new_betas.items()
        },
        organizations_to_include=organizations_to_include,
        use_cohort_names=use_cohort_names,
    )

    deviance = {}
    for cohort in cohort_names:
        cohort_partials = [result[cohort] for result in deviance_partials]
        deviance[cohort] = _compute_deviance(cohort_partials)
    info(" - Deviance computed!")

    return False, new_betas, deviance, cohort_names


def _filter_df_on_cohort_names(dfs, cohort_names, use_cohort_names):
    """
    Filter the dfs and cohort_names to only include the ones in use_cohort_names. This
    method orders the dfs and cohort_names to match the order of the use_cohort_names.
    """
    filtered_dfs = []
    filtered_cohort_names = []

    cohort_to_df = dict(zip(cohort_names, dfs))
    for cohort_name in use_cohort_names:
        filtered_dfs.append(cohort_to_df[cohort_name])
        filtered_cohort_names.append(cohort_name)

    return filtered_dfs, filtered_cohort_names


def _compute_central_betas(
    partial_betas: list[dict],
    family: str,
) -> dict:
    """
    Compute the central beta coefficients from the partial beta coefficients

    Parameters
    ----------
    partial_betas : list[dict]
        The partial beta coefficients from the nodes
    family : str
        The family of the GLM

    Returns
    -------
    dict
        A dictionary containing the central beta coefficients and related metadata
    """
    # sum the contributions of the partial betas

    info(f"partial_betas: {partial_betas}")

    info("Summing contributions of partial betas")
    total_observations = sum([partial["num_observations"] for partial in partial_betas])
    sum_observations = sum([partial["sum_y"] for partial in partial_betas])

    y_average = sum_observations / total_observations

    XTX_sum = reduce(
        lambda x, y: x + y, [pd.DataFrame(partial["XTX"]) for partial in partial_betas]
    )
    XTz_sum = reduce(
        lambda x, y: x + y, [pd.DataFrame(partial["XTz"]) for partial in partial_betas]
    )
    dispersion_sum = sum(
        [
            partial["dispersion"]
            for partial in partial_betas
            if partial["dispersion"] is not None
        ]
    )
    num_observations = sum([partial["num_observations"] for partial in partial_betas])
    # TODO is this always correct? What if one of the categorical predictors has
    # different levels between parties?
    num_variables = partial_betas[0]["num_variables"]

    if family == Family.GAUSSIAN.value:
        dispersion = dispersion_sum / (num_observations - num_variables)
        is_dispersion_estimated = True
    else:
        dispersion = 1
        is_dispersion_estimated = False

    info("Updating betas")

    XTX_np = XTX_sum.to_numpy()
    XTz_np = XTz_sum.to_numpy()

    beta_estimates = np.linalg.solve(XTX_np, XTz_np).flatten()
    std_error_betas = np.sqrt(np.diag(np.linalg.inv(XTX_np) * dispersion))

    # add the indices back to the beta estimates
    indices = pd.DataFrame(partial_betas[0]["XTX"]).index
    info(f"Indices: {indices}")
    info(f"Beta estimates: {beta_estimates}")
    beta_estimates = pd.Series(beta_estimates, index=indices)
    std_error_betas = pd.Series(std_error_betas, index=indices)

    return {
        "beta_estimates": beta_estimates.to_dict(),
        "std_error_betas": std_error_betas.to_dict(),
        "dispersion": dispersion,
        "is_dispersion_estimated": is_dispersion_estimated,
        "num_observations": num_observations,
        "num_variables": num_variables,
        "y_average": y_average,
    }


def _compute_deviance(
    partial_deviances: list[dict],
) -> dict:
    """
    Compute the total deviance from the partial deviances

    Parameters
    ----------
    partial_deviances : list[dict]
        The partial deviances from the nodes

    Returns
    -------
    dict
        A dictionary containing the total deviance for the null, old, and new models
    """
    total_deviance_null = sum(
        [partial["deviance_null"] for partial in partial_deviances]
    )
    total_deviance_old = sum([partial["deviance_old"] for partial in partial_deviances])
    total_deviance_new = sum([partial["deviance_new"] for partial in partial_deviances])
    return {
        "null": total_deviance_null,
        "old": total_deviance_old,
        "new": total_deviance_new,
    }


def _compute_local_betas_task(
    client: AlgorithmClient,
    formula: str,
    family: str,
    categorical_predictors: list[str],
    survival_sensor_column: str,
    iter_num: int,
    organizations_to_include: list[int],
    betas: list[int] | None = None,
    use_cohort_names: list[str] = [],
) -> list[dict]:
    """
    Create a subtask to compute the partial beta coefficients for each organization
    involved in the task

    Parameters
    ----------
    client : AlgorithmClient
        The client object to interact with the server
    formula : str
        The formula to use for the GLM
    family : str
        The family of the GLM
    categorical_predictors : list[str]
        The column names of the predictor variables to be treated as categorical
    survival_sensor_column : str
        The survival_sensor_column value
    iter_num : int
        The iteration number
    organizations_to_include : list[int]
        The organizations to include in the computation
    betas : list[int], optional
        The beta coefficients from the previous iteration, by default None
    use_cohort_names : list[str], optional
        The cohort names to include in the computation, by default []

    Returns
    -------
    list[dict]
        The results of the subtask
    """
    info("Defining input parameters")
    input_ = {
        "method": "compute_local_betas",
        "kwargs": {
            "use_cohort_names": use_cohort_names,
            "formula": formula,
            "family": family,
            "is_first_iteration": iter_num == 1,
        },
    }
    if categorical_predictors:
        input_["kwargs"]["categorical_predictors"] = categorical_predictors
    if survival_sensor_column:
        input_["kwargs"]["survival_sensor_column"] = survival_sensor_column
    if betas:
        input_["kwargs"]["beta_coefficients"] = betas

    # create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        input_=input_,
        organizations=organizations_to_include,
        name="Partial betas subtask",
        description=f"Subtask to compute partial betas - iteration {iter_num}",
    )

    # wait for node to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")

    # check that each node provided complete results
    _check_partial_results(
        results,
        ["XTX", "XTz", "dispersion", "num_observations", "num_variables", "sum_y"],
    )

    return results


def _compute_partial_deviance(
    client: AlgorithmClient,
    formula: str,
    family: str,
    categorical_predictors: list[str] | None,
    iter_num: int,
    survival_sensor_column: str,
    beta_estimates: pd.Series,
    beta_estimates_previous: pd.Series | None,
    global_average_outcome_var: list[int],
    organizations_to_include: list[int],
    use_cohort_names: list[str],
) -> list[dict]:
    """
    Create a subtask to compute the partial deviance for each organization involved in
    the task

    Parameters
    ----------
    client : AlgorithmClient
        The client object to interact with the server
    formula : str
        The formula to use for the GLM
    family : str
        The family of the GLM
    categorical_predictors : list[str] | None
        The column names of the predictor variables to be treated as categorical
    iter_num : int
        The iteration number
    survival_sensor_column : str
        The survival_sensor_column value
    beta_estimates : pd.Series
        The beta coefficients from the current iteration
    beta_estimates_previous : pd.Series | None
        The beta coefficients from the previous iteration
    global_average_outcome_var : int
        The global average of the outcome variable
    organizations_to_include : list[int]
        The organizations to include in the computation
    use_cohort_names : list[str]
        The cohort names to include in the computation

    Returns
    -------
    dict
        The results of the subtask
    """
    info("Defining input parameters")
    input_ = {
        "method": "compute_local_deviance",
        "kwargs": {
            "formula": formula,
            "family": family,
            "is_first_iteration": iter_num == 1,
            "beta_coefficients": beta_estimates,
            "global_average_outcome_var": global_average_outcome_var,
            "use_cohort_names": use_cohort_names,
        },
    }
    if categorical_predictors:
        input_["kwargs"]["categorical_predictors"] = categorical_predictors
    if survival_sensor_column:
        input_["kwargs"]["survival_sensor_column"] = survival_sensor_column
    if beta_estimates_previous:
        input_["kwargs"]["beta_coefficients_previous"] = beta_estimates_previous

    # create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        input_=input_,
        organizations=organizations_to_include,
        name="Partial deviance subtask",
        description=f"Subtask to compute partial deviance - iteration {iter_num}",
    )

    # wait for node to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")

    # check that each node provided complete results
    _check_partial_results(results, ["deviance_null", "deviance_old", "deviance_new"])

    return results


def _check_partial_results(results: list[dict], required_keys: list[str]) -> None:
    """
    Check that each of the partial results contains complete data
    """
    for result_node in results:
        for cohort_name, result in result_node.items():
            if result is None:
                raise AlgorithmExecutionError(
                    f"At least one of the nodes returned invalid result for cohort {cohort_name}. Please check the "
                    "logs."
                )
            for key in required_keys:
                if key not in result:
                    raise AlgorithmExecutionError(
                        f"At least one of the nodes returned incomplete result for cohort {cohort_name}. Please check"
                        " the logs."
                    )


def _check_input(
    organizations_to_include: list[int],
    family: str,
    formula: str | None,
    outcome_variable: str | None,
    predictor_variables: list[str] | None,
    survival_sensor_column: str | None,
) -> None:
    """
    Check that the input is valid

    Parameters
    ----------
    organizations_to_include : list[int]
        The organizations to include in the computation
    family : str
        The family of the GLM
    formula : str | None
        The formula to use for the GLM
    outcome_variable : str | None
        The name of the outcome variable column
    predictor_variables : list[str] | None
        The names of the predictor variable columns
    survival_sensor_column : str | None
        The survival_sensor_column value

    Raises
    ------
    UserInputError
        If the input is invalid
    """
    if not organizations_to_include:
        raise UserInputError("No organizations provided in the input.")

    min_orgs = get_env_var(
        ENVVAR_MINIMUM_ORGANIZATIONS, DEFAULT_MINIMUM_ORGANIZATIONS, as_type="int"
    )
    if len(organizations_to_include) < min_orgs:
        raise UserInputError(
            "Number of organizations included in the computation is less than the "
            f"minimum required ({min_orgs})."
        )

    # Either formula or outcome and predictor variables should be provided
    if formula and (outcome_variable or predictor_variables):
        warn(
            "Both formula or outcome and predictor variables are provided - using "
            "the formula and ignoring the outcome/predictor."
        )
    if not formula and not (outcome_variable and predictor_variables):
        raise UserInputError(
            "Either formula or outcome and predictor variables should be provided. "
            "Neither is provided."
        )

    if family == Family.SURVIVAL.value and not survival_sensor_column:
        raise UserInputError(
            "The survival family requires the survival_sensor_column to be provided."
        )


def _log_header(num_iteration: int) -> None:
    """
    Print header for the iteration to the logs
    """
    info("")
    info("#" * 60)
    info(f"# Starting iteration {num_iteration}")
    info("#" * 60)
    info("")


def _compute_local_betas(
    df: pd.DataFrame,
    formula: str,
    family: str,
    is_first_iteration: bool,
    beta_coefficients: dict[str, float] | None = None,
    categorical_predictors: list[str] | None = None,
    survival_sensor_column: str = None,
) -> dict:
    """
    Compute beta coefficients for a GLM model

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    formula : str
        The formula specifying the model.
    family : str
        The family of the GLM (e.g., 'gaussian', 'binomial').
    is_first_iteration : bool
        Whether this is the first iteration of the model.
    beta_coefficients : dict[str, float] | None
        The beta coefficients. These must be provided if is_first_iteration is False.
    categorical_predictors : list[str] | None
        Predictor variables that should be treated as categorical.
    survival_sensor_column : str, optional
        An optional parameter for additional model specifications.

    Returns
    -------
    dict
        The computed beta coefficients.
    """
    info("Started function to compute beta coefficients")

    # convert input dicts to pandas
    if beta_coefficients is not None:
        beta_coefficients = pd.Series(beta_coefficients)

    data_mgr = GLMDataManager(
        df,
        formula,
        family,
        categorical_predictors,
        survival_sensor_column,
    )
    y_column_names = data_mgr.y.columns
    info(f"y_column_names: {y_column_names}")

    eta = data_mgr.compute_eta(is_first_iteration, beta_coefficients)

    info("Computing beta coefficients")
    mu = data_mgr.compute_mu(eta, y_column_names)
    info(f"mu: {mu}")
    varg = data_mgr.family.variance(mu)
    info(f"varg: {varg}")
    varg = cast_to_pandas(varg, columns=y_column_names)

    # TODO in R, we can do gprime <- family$mu.eta(eta), but in Python I could not
    # find a similar function. It is therefore now implemented for each family
    if isinstance(data_mgr.family, families.Poisson):
        # for poisson, this is exp(eta)
        gprime = data_mgr.family.link.inverse(eta)
    elif isinstance(data_mgr.family, families.Binomial):
        # for binomial, this is mu * (1 - mu), which is the same as the variance func
        gprime = data_mgr.family.variance(mu)
    else:
        # For Gaussian family
        gprime = data_mgr.family.link.deriv(eta)
    gprime = cast_to_pandas(gprime, columns=y_column_names)

    # compute Z matrix and dispersion matrix
    y_minus_mu = data_mgr.y.sub(mu.values, axis=0)

    print("dimensions of y_minus_mu", y_minus_mu.shape)
    print("dimensions of mu", mu.shape)
    print("dimensions of eta", eta.shape)

    z = eta + (y_minus_mu.values / gprime.values)

    print("dimensions of z", z.shape)
    print("dimensions of gprime", gprime.shape)
    print(
        "dimensions of y_minus_mu / gprime", (y_minus_mu.values / gprime.values).shape
    )
    print("y-columns", y_column_names)
    print("gprime-columns", gprime.columns)
    print("y_minus_mu-columns", y_minus_mu.columns)

    W = gprime**2 / varg

    if family == Family.GAUSSIAN.value:
        dispersion_matrix = W * (y_minus_mu / gprime) ** 2
        dispersion = float(dispersion_matrix.sum().iloc[0])
    else:
        # For non-Gaussian families, the dispersion is not estimated so we don't need to
        # share any information about it.
        dispersion = None

    _check_privacy(df, len(data_mgr.X.columns))

    # TODO there are some non-clear things in the code like `mul()` and `iloc[:, 0]`.
    # They are there to ensure proper multiplication etc of pandas Dataframes with
    # series. Make this code more clear and readable.

    print("dimensions of X", data_mgr.X.shape)
    print("dimensions of W", W.shape)
    print("dimensions of z", z.shape)

    print(f"W.iloc[:, 0] shape: {W.iloc[:, 0].shape}")
    # print(f"X.mul result shape: {data_mgr.X.mul(W.iloc[:, 0], axis=0).shape}")
    print(f"X.T shape: {data_mgr.X.T.shape}")

    print(f"X index: {data_mgr.X.index}")
    print(f"W index: {W.index}")
    # Use numpy arrays directly to avoid index alignment issues
    X_weighted = (
        data_mgr.X.values * W.values
    )  # Broadcasting will handle the multiplication
    XT_dot_X_times_W = data_mgr.X.values.T.dot(X_weighted)
    XT_dot_W_times_z = data_mgr.X.values.T.dot(W.values * z.values)

    XTX_df = pd.DataFrame(
        XT_dot_X_times_W,
        index=data_mgr.X.columns,  # Variable names as index
        columns=data_mgr.X.columns,  # Variable names as columns
    )

    XTz_df = pd.DataFrame(
        XT_dot_W_times_z,
        index=data_mgr.X.columns,  # Variable names as index
        columns=y_column_names,  # Response variable name as columns
    )

    return {
        "XTX": XTX_df.to_dict(),
        "XTz": XTz_df.to_dict(),
        "dispersion": dispersion,
        "num_observations": len(df),
        "num_variables": len(data_mgr.X.columns),
        "sum_y": float(data_mgr.y.sum().iloc[0]),
    }


def _check_privacy(df: pd.DataFrame, num_variables: int):
    """
    Check that the privacy threshold is not violated.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    num_variables : int
        The number of variables in the model.

    Raises
    ------
    PrivacyThresholdViolation
        If the privacy threshold is violated.
    """
    # before returning the data, check that the model has limited risks of overfitting.
    # If too many variables are used, there is a chance the data will be reproducible.
    # This is a security measure to prevent data leakage.
    max_pct_vars_vs_obs = get_env_var(
        ENVVAR_MAX_PCT_PARAMS_OVER_OBS, DEFAULT_MAX_PCT_PARAMS_VS_OBS, as_type="int"
    )
    if num_variables * 100 / len(df) > max_pct_vars_vs_obs:
        raise PrivacyThresholdViolation(
            "Number of variables is too high compared to the number of observations. "
            f"This is not allowed to be more than {max_pct_vars_vs_obs}% but is "
            f"{num_variables * 100 / len(df)}%."
        )


class Family(str, Enum):
    """TODO docstring"""

    # TODO add more families. Available from statsmodels.genmod.families:
    # from .family import Gaussian, Family, Poisson, Gamma, \
    #     InverseGaussian, Binomial, NegativeBinomial, Tweedie
    POISSON = "poisson"
    BINOMIAL = "binomial"
    GAUSSIAN = "gaussian"
    SURVIVAL = "survival"


# TODO integrate with enum
def get_family(family: str) -> Family:
    """TODO docstring"""
    # TODO figure out which families are supported
    # TODO use survival_sensor_column?
    if family == Family.POISSON.value:
        return sm.families.Poisson()
    elif family == Family.BINOMIAL.value:
        return sm.families.Binomial()
    elif family == Family.GAUSSIAN.value:
        return sm.families.Gaussian()
    elif family == Family.SURVIVAL.value:
        return sm.families.Poisson()
    else:
        raise UserInputError(
            f"Family {family} not supported. Please provide one of the supported "
            f"families: {', '.join([fam.value for fam in Family])}"
        )


def get_formula(
    outcome_variable: str,
    predictor_variables: list[str],
    category_reference_variables: list[str],
    categorical_predictors: list[str] | None = None,
) -> str:
    """
    Get the formula for the GLM model from the outcome and predictor variables.

    If category_reference_variables is provided, the formula will be created with
    these variables as reference categories according to the formulaic package's
    syntax.

    Parameters
    ----------
    outcome_variable : str
        The outcome variable
    predictor_variables : list[str]
        The predictor variables
    category_reference_variables : list[str]
        The reference categories for the predictor variables
    categorical_predictors : list[str] | None
        Predictor variables that should be treated as categorical even though they are
        numerical.

    Returns
    -------
    str
        The formula for the GLM model
    """
    predictors = {}
    if category_reference_variables is not None:
        for var in predictor_variables:
            if var in category_reference_variables:
                ref_value = category_reference_variables[var]
                if (
                    categorical_predictors is None
                    or var not in categorical_predictors
                    or isinstance(ref_value, str)
                ):
                    ref_value = f"'{ref_value}'"
                predictors[var] = f"C({var}, Treatment(reference={ref_value}))"
            else:
                predictors[var] = var
    else:
        predictors = {var: var for var in predictor_variables}
    return f"{outcome_variable} ~ {' + '.join(predictors.values())}"


def cast_to_pandas(
    data_: np.ndarray | pd.Series | pd.DataFrame | Any,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Cast a numpy array to a pandas Series.

    Parameters
    ----------
    data : np.ndarray | pd.Series | pd.DataFrame
        The data to cast. This function does nothing if the data is not a numpy array.
    columns : list[str] | None
        The column names to give in the resulting pandas Data frame

    Returns
    -------
    pd.Series
        The data as a pandas Series.
    """
    if isinstance(data_, np.ndarray):
        info("Casting numpy array to pandas DataFrame...")
        return pd.DataFrame(data_.flatten(), columns=columns)
    return pd.DataFrame(data_, columns=columns)


class GLMDataManager:
    """
    A class to manage data for Generalized Linear Models (GLM).

    Attributes
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    formula : str
        The formula specifying the model.
    family_str : str
        The family of the GLM (e.g., 'gaussian', 'binomial').
    survival_sensor_column : str, optional
        An optional parameter for additional model specifications.
    y : pd.Series
        The response variable.
    X : pd.DataFrame
        The design matrix.
    family : Family
        The family object corresponding to the family_str.
    mu_start : pd.Series or None
        The initial values for the mean response.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        formula: str,
        family: str,
        categorical_predictors: list[str] | None,
        survival_sensor_column: str = None,
    ) -> None:
        """
        Initialize the GLMDataManager.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the data.
        formula : str
            The formula specifying the model.
        family : str
            The family of the GLM (e.g., 'gaussian', 'binomial').
        categorical_predictors : list[str] | None
            Predictor variables that should be treated as categorical.
        survival_sensor_column : str, optional
            An optional parameter for additional model specifications.
        """

        self.df = df
        self.formula = formula
        self.family_str = family
        self.survival_sensor_column = survival_sensor_column

        # User can indicate if there are numerical predictors that should be treated as
        # categorical.
        if categorical_predictors is not None:
            for predictor in categorical_predictors:
                self.df[predictor] = self.df[predictor].astype("category")

        self.y, self.X = self._get_design_matrix()
        self.y = cast_to_pandas(self.y)
        self.X = cast_to_pandas(self.X)

        self.family = get_family(self.family_str)

        self.mu_start: pd.Series | None = None

        self._privacy_checks()

    def compute_eta(
        self, is_first_iteration: bool, betas: pd.Series | None
    ) -> pd.Series:
        """
        Compute the eta values for the GLM model.

        Parameters
        ----------
        is_first_iteration : bool
            Whether this is the first iteration of the model.
        betas : pd.Series | None
            The beta coefficients. These must be provided if is_first_iteration is
            False.

        Returns
        -------
        pd.Series
            The eta values for the GLM model.
        """
        info("Computing eta values")
        if is_first_iteration:
            if self.mu_start is None:
                self.set_mu_start()
            if self.family_str == Family.SURVIVAL:
                survival_sensor_column = self.df[self.survival_sensor_column]
                eta = (self.mu_start.squeeze() - survival_sensor_column).apply(np.log)
                eta = cast_to_pandas(eta)
            else:
                eta = self.family.link(self.mu_start)
        else:
            # dot product cannot be done with a series, so convert to numpy array and
            # reshape to get betas in correct format
            betas = betas.values.reshape(-1, 1)
            eta = self.X.dot(betas)
        eta.columns = self.y.columns
        return eta

    def compute_mu(self, eta: pd.Series, columns: list[str] | None = None) -> pd.Series:
        """
        Compute the mean response variable for the GLM model.

        Parameters
        ----------
        eta : pd.Series
            The eta values.
        columns : list[str] | None
            The column names of the response variable. Optional.

        Returns
        -------
        pd.Series
            The mean response variable.
        """
        if self.family_str == Family.SURVIVAL:
            # custom link function for survival models
            mu = self.df[self.survival_sensor_column].add(eta.squeeze().apply(np.exp))
        else:
            mu = self.family.link.inverse(eta)
        return cast_to_pandas(mu, columns=columns)

    def compute_deviance(self, mu: pd.Series) -> float:
        """
        Compute the deviance for the GLM model.

        Parameters
        ----------
        mu : pd.Series
            The mean response variable.

        Returns
        -------
        float
            The deviance for the GLM model.
        """
        y = self.y.squeeze()
        if isinstance(mu, pd.DataFrame):
            mu = mu.squeeze()
        return self.family.deviance(y, mu)

    def set_mu_start(self) -> None:
        """
        Set the initial values for the mean response variable.
        """
        if self.family_str == Family.SURVIVAL:
            self.mu_start = (
                np.maximum(self.y.squeeze(), self.df[self.survival_sensor_column]) + 0.1
            )
            self.mu_start = cast_to_pandas(self.mu_start)
        else:
            self.mu_start = self.family.starting_mu(self.y)

    def _get_design_matrix(self) -> tuple[pd.Series, pd.DataFrame]:
        """
        Create the design matrix X and predictor variable y

        Returns
        -------
        Tuple[pd.Series, pd.DataFrame]
            A tuple containing the predictor variable y and the design matrix X
        """
        info("Creating design matrix X and predictor variable y")
        y, X = Formula(self.formula).get_model_matrix(self.df)
        X.columns = self._simplify_column_names(X.columns)
        return y, X

    def _privacy_checks(self) -> None:
        """
        Do privacy checks on the data after initializing the GLMDataManager.

        Raises
        ------
        PrivacyThresholdViolation
            If the data contains too few values for at least one category of a
            categorical variable.
        """
        # check if dataframe is long enough
        min_rows = get_env_var(
            ENVVAR_MINIMUM_ROWS, default=DEFAULT_MINIMUM_ROWS, as_type="int"
        )
        if len(self.df) < min_rows:
            raise PrivacyThresholdViolation(
                f"Data contains less than {min_rows} rows. Refusing to "
                "handle this computation, as it may lead to privacy issues."
            )

        # check which columns the formula needs. These require some additional checks
        columns_used = Formula(self.formula).required_variables

        # check that a column has at least required number of non-null values
        for col in columns_used:
            if self.df[col].count() < min_rows:
                raise PrivacyThresholdViolation(
                    f"Column {col} contains less than {min_rows} non-null values. "
                    "Refusing to handle this computation, as it may lead to privacy "
                    "issues."
                )

        # Check if requested columns are allowed to be used for GLM by node admin
        allowed_columns = get_env_var(ENVVAR_ALLOWED_COLUMNS)
        if allowed_columns:
            allowed_columns = allowed_columns.split(",")
            for col in columns_used:
                if col not in allowed_columns:
                    raise NodePermissionException(
                        f"The node administrator does not allow '{col}' to be requested"
                        " in this algorithm computation. Please contact the node "
                        "administrator for more information."
                    )
        non_allowed_collumns = get_env_var(ENVVAR_DISALLOWED_COLUMNS)
        if non_allowed_collumns:
            non_allowed_collumns = non_allowed_collumns.split(",")
            for col in columns_used:
                if col in non_allowed_collumns:
                    raise NodePermissionException(
                        f"The node administrator does not allow '{col}' to be requested"
                        " in this algorithm computation. Please contact the node "
                        "administrator for more information."
                    )

    @staticmethod
    def _simplify_column_names(columns: pd.Index) -> pd.Index:
        """
        Simplify the column names of the design matrix

        Parameters
        ----------
        columns : pd.Index
            The column names of the design matrix
        predictors : list[str]
            The predictor variables

        Returns
        -------
        pd.Index
            The simplified column names
        """
        # remove the part of the column name that specifies the reference value
        # e.g. C(prog, Treatment(reference='General'))[T.Vocational] ->
        # prog[T.Vocational]
        pattern = r"C\(([^,]+), Treatment\(reference=[^\)]+\)\)\[([^\]]+)\]"
        replacement = r"\1[\2]"
        simplified_columns = [
            re.sub(pattern, replacement, column_name) for column_name in columns
        ]
        return pd.Index(simplified_columns)


def _compute_local_deviance(
    df: pd.DataFrame,
    formula: str,
    family: str,
    is_first_iteration: bool,
    global_average_outcome_var: float,
    beta_coefficients: list[int],
    beta_coefficients_previous: list[int] | None = None,
    categorical_predictors: list[str] | None = None,
    survival_sensor_column: str | None = None,
) -> dict:
    """
    Compute the local deviance for a GLM model given the beta coefficients of the global
    model.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    formula : str
        The formula specifying the model.
    family : str
        The family of the GLM (e.g., 'gaussian', 'binomial').
    is_first_iteration : bool
        Whether this is the first iteration of the model.
    global_average_outcome_var : float
        The global average of the response variable.
    beta_coefficients : list[int]
        The beta coefficients of the current model.
    beta_coefficients_previous : list[int]
        The beta coefficients of the previous model.
    categorical_predictors : list[str] | None
        Predictor variables that should be treated as categorical.
    survival_sensor_column : str | None
        An optional parameter for additional model specifications.

    Returns
    -------
    dict
        The computed deviance values.
    """
    # TODO this function computes deviance_null which is never used. Why?
    info("Computing local deviance")

    data_mgr = GLMDataManager(
        df,
        formula,
        family,
        categorical_predictors,
        survival_sensor_column,
    )

    beta_coefficients = pd.Series(beta_coefficients)
    beta_coefficients_previous = pd.Series(beta_coefficients_previous)

    # update mu and compute deviance, then compute eta
    eta_old = data_mgr.compute_eta(is_first_iteration, beta_coefficients_previous)
    if is_first_iteration:
        data_mgr.set_mu_start()
        mu_old = data_mgr.mu_start
        deviance_old = 0
    else:
        mu_old = data_mgr.compute_mu(eta_old)
        deviance_old = data_mgr.compute_deviance(mu_old)

    # update beta coefficients
    eta_new = data_mgr.compute_eta(is_first_iteration=False, betas=beta_coefficients)
    mu_new = data_mgr.compute_mu(eta_new)

    deviance_new = data_mgr.compute_deviance(mu_new)
    # TODO deviance null is the same every cycle - maybe not compute every time. On the
    # other hand, it is fast and easy and this way code is easier to understand
    deviance_null = data_mgr.compute_deviance(global_average_outcome_var)

    return {
        "deviance_old": float(deviance_old),
        "deviance_new": float(deviance_new),
        "deviance_null": float(deviance_null),
    }
