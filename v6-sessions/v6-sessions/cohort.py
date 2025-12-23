import traceback

import numpy as np
import pandas as pd
import pkg_resources
import pyarrow as pa
from ohdsi import common, database_connector, sqlrender
from rpy2.robjects import RS4
from rpy2.rinterface_lib.sexp import NACharacterType
from vantage6.algorithm.decorator import data_extraction
from vantage6.algorithm.tools.util import error, info


@data_extraction
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
        dbms="postgresql",
        connection_string=connection_details["uri"],
        user=connection_details["USER"],
        password=connection_details["PASSWORD"],
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
        # Manually construct the VALUES clause for patient IDs
        values_clause = ", ".join([f"({pid})" for pid in patient_ids])
        raw_sql = tuple(raw_sql)[0]
        rendered_sql = raw_sql.replace("{@patient_ids}", values_clause)
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
        with open("errorReportSql.txt", "r") as f:
            error(f.read())

        raise e

    info("Converting dataframe to pandas")
    try:
        converted_df = common.convert_from_r(df, date_cols=["surgery_date"])
    except Exception as e:
        error(f"Failed to convert dataframe: {e}")
        traceback.print_exc()
        raise e

    converted_df = converted_df.applymap(
        lambda val: np.nan if isinstance(val, NACharacterType) else val
    )

    # Somehow the dataframe is missing some metadata, so we need to create a new
    # dataframe with the same data and the same columns.
    clean_df = pd.DataFrame(converted_df.values, columns=converted_df.columns)
    
    # All column names to lowercase
    clean_df.columns = clean_df.columns.str.lower()

    # DROP DUPLICATES
    sub_df = clean_df.drop_duplicates("patient_id", keep="first")
    info(f"Dropped {len(clean_df) - len(sub_df)} rows")

    info("Converting column types")
    # Numeric columns
    sub_df["patient_id"] = pd.to_numeric(sub_df["patient_id"], errors="coerce")
    sub_df["age"] = pd.to_numeric(sub_df["age"], errors="coerce")
    sub_df["survival_days"] = pd.to_numeric(sub_df["survival_days"], errors="coerce")
    sub_df["tumor_size"] = pd.to_numeric(sub_df["tumor_size"], errors="coerce")
    sub_df["surgery_concept"] = pd.to_numeric(sub_df["surgery_concept"], errors="coerce")
    sub_df["completeness_of_resection_concept_id"] = pd.to_numeric(sub_df["completeness_of_resection_concept_id"], errors="coerce")
    sub_df["n_cancer_episodes"] = pd.to_numeric(sub_df["n_cancer_episodes"], errors="coerce")
    
    # Boolean columns (CASE statements that return 1/0)
    sub_df["censor"] = sub_df["censor"].astype("bool")
    sub_df["tumor_rupture"] = sub_df["tumor_rupture"].astype("bool")
    sub_df["pre_operative_chemo"] = sub_df["pre_operative_chemo"].astype("bool")
    sub_df["post_operative_chemo"] = sub_df["post_operative_chemo"].astype("bool")
    sub_df["pre_operative_radio"] = sub_df["pre_operative_radio"].astype("bool")
    sub_df["post_operative_radio"] = sub_df["post_operative_radio"].astype("bool")
    sub_df["local_recurrence"] = sub_df["local_recurrence"].astype("bool")
    sub_df["distant_metastasis"] = sub_df["distant_metastasis"].astype("bool")
    
    # Category columns
    sub_df["sex"] = sub_df["sex"].astype("category")
    sub_df["status"] = sub_df["status"].astype("category")
    sub_df["histology"] = sub_df["histology"].astype("category")
    sub_df["fnclcc_grade"] = sub_df["fnclcc_grade"].astype("category")
    sub_df["multifocality"] = sub_df["multifocality"].astype("category")
    sub_df["completeness_of_resection"] = sub_df["completeness_of_resection"].astype("category")
    
    # Datetime columns
    sub_df["surgery_date"] = pd.to_datetime(sub_df["surgery_date"], errors="coerce", utc=True).dt.normalize()

    info("-->  Done")

    return pa.Table.from_pandas(sub_df)
