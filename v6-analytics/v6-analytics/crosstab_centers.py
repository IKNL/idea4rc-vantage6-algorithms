import os
from functools import wraps

import pandas as pd
from scipy import stats

from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.decorator import metadata, dataframes
from vantage6.algorithm.decorator.metadata import RunMetaData
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
from vantage6.algorithm.tools.util import error
from vantage6.common.globals import ContainerEnvNames

def _algorithm_client() -> callable:
    def protection_decorator(func: callable, *args, **kwargs) -> callable:
        @wraps(func)
        def decorator(
            *args, mock_client: MockAlgorithmClient | None = None, **kwargs
        ) -> callable:
            """
            Wrap the function with the client object

            Parameters
            ----------
            mock_client : MockAlgorithmClient | None
                Mock client. If not None, used instead of the regular client
            """
            if mock_client is not None:
                return func(mock_client, *args, **kwargs)

            # read token from the environment
            token = os.environ.get(ContainerEnvNames.CONTAINER_TOKEN.value)
            if not token:
                error(
                    "Token not found. Is the method you called started as a "
                    "compute container? Exiting..."
                )
                exit(1)

            # read server address from the environment
            host = os.environ[ContainerEnvNames.HOST.value]
            port = os.environ[ContainerEnvNames.PORT.value]
            api_path = os.environ[ContainerEnvNames.API_PATH.value]

            client = AlgorithmClient(
                token=token,
                server_url=f"{host}:{port}{api_path}",
                auth_url="does-not-matter",
            )
            return func(client, *args, **kwargs)

        # set attribute that this function is wrapped in an algorithm client
        decorator.wrapped_in_algorithm_client_decorator = True
        return decorator

    return protection_decorator


algorithm_client = _algorithm_client()


@algorithm_client
def crosstab_centers(
    client: AlgorithmClient,
    organizations_to_include: list[int] | None = None,
) -> tuple[dict, dict]:
    """ """
    organizations = client.organization.list()
    if not organizations_to_include:
        organizations_to_include = [
            organization.get("id") for organization in organizations
        ]

    # Get the data from the centers
    task = client.task.create(
        method="compute_local_counts",
        input_={},
        organizations=organizations_to_include,
        name="Cross tabulation centers subtask",
        description="Subtask to compute cross tabulation centers",
    )

    # Wait for the task to finish
    results = client.wait_for_results(task_id=task.get("id"))

    # Combine the results
    combined_df, chi_squared_df = combine_center_results(results, organizations)

    return combined_df, chi_squared_df


def combine_center_results(
    center_results: list[dict[str, list[dict[str, dict[str, int]]]]],
    organizations: list[dict],
) -> tuple[dict, dict]:
    """
    Combine results from multiple centers and compute chi-squared tests.

    This function assumes that all centers have the same cohorts and variables.

    Parameters
    ----------
    center_results : list
        List of dictionaries containing counts from each center

    Returns
    -------
    tuple
        (combined_counts, chi_squared_results) where:
        - combined_counts: DataFrame with counts from all centers
        - chi_squared_results: DataFrame with chi-squared test results
    """
    # For each cohort, for each variable, for each level, add the counts
    # We want to construct a dictionary with the levels as keys and the counts
    # as values:
    # [
    #     {
    #         "Cohort": cohort,
    #         "Variable": var,
    #         "Level": level,
    #         "organization_name_1": count,
    #         "organization_name_2": count,
    #         ...
    #     },
    #     ...
    # ]
    rows = []
    cohorts_names = [key for key in center_results[0].keys() if key != "meta"]
    for cohort in cohorts_names:

        variable_names = [key for key in center_results[0][cohort].keys()]
        for var in variable_names:
            all_levels = {
                level
                for center in center_results
                for level in center[cohort][var].keys()
            }

            for level in all_levels:
                counts = {}
                for center in center_results:
                    org_name = next(
                        org["name"]
                        for org in organizations
                        if org["id"] == center["meta"]["organization_id"]
                    )

                    cohort_center_counts = center[cohort]
                    counts[org_name] = cohort_center_counts[var].get(level, 0)

                rows.append(
                    {
                        "Cohort": cohort,
                        "Variable": var,
                        "Level": level,
                        **counts,
                    }
                )

    combined_df = pd.DataFrame(rows).sort_values(["Cohort", "Variable", "Level"])

    # Compute chi-squared tests
    organization_names = [org["name"] for org in organizations]
    center_cols = [col for col in combined_df.columns if col in organization_names]
    results = []

    for cohort in combined_df["Cohort"].unique():
        cohort_data = combined_df[combined_df["Cohort"] == cohort]
        for var in cohort_data["Variable"].unique():
            var_data = cohort_data[cohort_data["Variable"] == var]
            contingency_table = var_data[center_cols].values

            # Check if the contingency table has any zero rows/columns
            if (contingency_table.sum(axis=0) > 0).all() and (
                contingency_table.sum(axis=1) > 0
            ).all():
                try:
                    chi2, p_val = stats.chi2_contingency(contingency_table)[:2]
                except ValueError:
                    chi2, p_val = None, None
            else:
                chi2, p_val = None, None

            results.append([cohort, var, chi2, p_val])

    chi_squared_df = pd.DataFrame(
        results, columns=["Cohort", "Variable", "Chi-squared", "P-value"]
    ).sort_values(["Cohort", "Variable"])

    return combined_df.to_dict(), chi_squared_df.to_dict()


@metadata
@dataframes
def compute_local_counts(
    dataframes: dict[str, pd.DataFrame], meta: RunMetaData
) -> dict[str, list[dict[str, dict[str, int]]]]:
    """
    Compute local categorical value counts for each variable for multiple dataframes.

    Parameters
    ----------
    dataframes : dict[str, pandas.DataFrame]
        One or more dataframes containing the data
    meta : RunMetaData
        Metadata about the run, including organization information

    Returns
    -------
    dict[str, dict[str, dict[str, dict[str, int]]]]
        Nested dictionary with counts per cohort, variable and category level
        Structure:
        ```python
        {
            'meta': {
                'organization_id': int,
                'organization_name': str,
                'task_id': str,
                'task_name': str,
                'run_id': str,
                'run_name': str,
            },
            'cohort_name_1': {
                'VARIABLE_NAME': {'LEVEL_1': count, 'LEVEL_2': count},
                ...
            },
            'cohort_name_2': {
                'VARIABLE_NAME': {'LEVEL_1': count, 'LEVEL_2': count},
                ...
            }
            ...
        }
        ```
    """
    results = {}
    results["meta"] = {
        "node_id": meta.node_id,
        "organization_id": meta.organization_id,
    }
    for cohort, df in dataframes.items():
        variables = df.select_dtypes(include=["category", "object"]).columns
        results[cohort] = {}
        for var in variables:
            results[cohort][var] = df[var].value_counts().to_dict()
    return results
