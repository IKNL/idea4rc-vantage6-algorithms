import pandas as pd

from scipy import stats

from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client, metadata, RunMetaData

from .decorator import new_data_decorator


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
        input_={
            "method": "compute_local_counts",
        },
        organizations=organizations_to_include,
        name="Crosstab centers subtask",
        description=f"Subtask to compute crosstab centers",
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
            var_dict = center_results[0][cohort][var]

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

    return combined_df.to_json(), chi_squared_df.to_json()


@metadata
@new_data_decorator
def compute_local_counts(
    dfs: list[pd.DataFrame], cohort_names: list[str], meta: RunMetaData
) -> dict[str, list[dict[str, dict[str, int]]]]:
    """
    Compute local categorical value counts for each variable for multiple dataframes.

    Parameters
    ----------
    dfs : list[pandas.DataFrame]
        One or more dataframes containing the data
    meta : RunMetaData
        Metadata about the run, including organization information
    cohort_names : list[str]
        Names of the cohorts corresponding to each dataframe

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
    if len(dfs) != len(cohort_names):
        raise ValueError("Number of dataframes must match number of cohort names")

    results = {}
    results["meta"] = {
        "node_id": meta.node_id,
        "organization_id": meta.organization_id,
    }
    for df, cohort in zip(dfs, cohort_names):
        variables = df.select_dtypes(include=["category"]).columns
        results[cohort] = {}
        for var in variables:
            results[cohort][var] = df[var].value_counts().to_dict()
    return results
