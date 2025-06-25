import traceback
import pkg_resources

import pandas as pd

from rpy2.robjects import RS4
from vantage6.algorithm.tools.util import info, error
from vantage6.algorithm.decorator import source_database
from ohdsi import sqlrender
from ohdsi import database_connector


@source_database
def create_cohort(
    connection_details: dict, patient_ids: list[int], features: str
) -> pd.DataFrame:
    """
    This function creates a cohort from a list of patient IDs.

    Arguments
    ----------
    connection_details: dict
        The connection details for the database. It should contain the following keys:
        - uri: The URI of the database.
        - user: The username to connect to the database.
        - password: The password to connect to the database.
    patient_ids: list[int]
        The list of patient IDs to create the cohort from.
    features: str
        The 'sarcoma' or 'head_neck' features to use.
    Returns
    -------
    pd.DataFrame
        A dataframe with the cohort data.
    """

    info("Setting up connection to database")
    connection = database_connector.connect(
        connection_string=connection_details["uri"],
        user=connection_details["user"],
        password=connection_details["password"],
    )

    info(f"Retrieving variables for cohort: {patient_ids}")
    try:
        df = __create_cohort_dataframe(connection, patient_ids, features)

    except Exception as e:
        error(f"Failed to create cohort dataframe for {patient_ids}")
        traceback.print_exc()
        raise e

    info("Done!")
    return df


def __create_cohort_dataframe(
    connection: RS4, patient_ids: list[int], features: str
) -> pd.DataFrame:
    """
    This function creates a cohort dataframe from a list of patient IDs.
    """
    info(f"Loading SQL file: {features}")
    sql_path = pkg_resources.resource_filename(
        "v6-sessions",
        f"sql/{features}_features.sql",
    )
    try:
        raw_sql = sqlrender.read_sql(sql_path)
    except Exception as e:
        error(f"Failed to read SQL file: {e}")
        traceback.print_exc()
        raise e
    info("-->  Done")

    info("Injecting patient IDs into SQL")
    try:
        rendered_sql = sqlrender.render(raw_sql, patient_ids=patient_ids)
    except Exception as e:
        error(f"Failed to render SQL: {e}")
        traceback.print_exc()
        raise e
    info("-->  Done")

    # In IDEA4RC we only use PostgreSQL, so we do not need to translate the SQL
    # info("Translating the SQL")
    # sql = sqlrender.translate(sql, target_dialect="postgresql")

    info("Executing SQL")
    try:
        df = database_connector.query_sql(connection, rendered_sql)
    except Exception as e:
        error(f"Failed to execute SQL: {e}")
        traceback.print_exc()
        raise e
    info("-->  Done")

    return df
