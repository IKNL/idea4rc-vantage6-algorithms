import traceback
from importlib.resources import as_file, files

import numpy as np
import pandas as pd
import pyarrow as pa
from jinja2 import Template
from ohdsi import common, database_connector
from rpy2.rinterface_lib.sexp import NACharacterType
from rpy2.robjects import RS4
from v6_idea4rc_common import (
    to_boolean as _to_boolean,
    to_category as _to_category,
    to_datetime as _to_datetime,
    to_int64 as _to_int64,
)
from vantage6.algorithm.decorator import data_extraction
from vantage6.algorithm.tools.util import error, get_env_var, info


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

    Arguments
    ----------
    connection: RS4
        The connection to the database.
    patient_ids: list[int]
        The list of patient IDs to create the cohort from.
    features: str
        The 'sarcoma' or 'head_neck' features to use.
    Returns
    -------
    pd.DataFrame
        A dataframe with the cohort data.
    """

    info(f"Loading SQL file: {features}")
    ref = files("v6-sessions").joinpath("sql", "features.sql.j2")
    try:
        with as_file(ref) as sql_path:
            sql_template = Template(open(sql_path).read())
    except Exception as e:
        error(f"Failed to read SQL file: {e}")
        traceback.print_exc()
        raise e

    info("Loading environment variables")
    cdm_schema = get_env_var("CDM_SCHEMA", "cdm_idea")

    info("Rendering SQL template")
    rendered_sql = sql_template.render(
        patient_ids=", ".join([f"({pid})" for pid in patient_ids]),
        cdm_schema=cdm_schema,
        is_head_and_neck=(features == "head_and_neck"),
        is_sarcoma=(features == "sarcoma"),
    )

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
        converted_df = common.convert_from_r(df, date_cols=[
            "date_of_surgery",
            "diagnosis_date",
            "life_status_date",
            "date_of_biopsy",
            "last_contact",
        ])
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
    info(
        f"Dropped {len(clean_df) - len(sub_df)} rows because of duplicate patient IDs, "
        "keeping the first occurrence. If this number is not 0, please check the data "
        "or adjust the SQL query."
    )

    info(f"Number of rows in dataframe: {len(sub_df)}")

    # Once we checked that there are no duplicates, we can remove
    # the patient_id column.
    sub_df = sub_df.drop(columns=["patient_id"])
    info("Removed patient_id column from dataframe")

    info(f"Converting column types for features: {features}")
    sub_df = convert_base_columns(sub_df)
    if features == "head_and_neck":
        sub_df = convert_head_neck_columns(sub_df)
    elif features == "sarcoma":
        sub_df = convert_sarcoma_columns(sub_df)
    else:
        raise ValueError(f"Invalid features: {features}")

    info("-->  Done")

    return pa.Table.from_pandas(sub_df)


def convert_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function converts the base columns of the dataframe to the correct types.
    """

    df = _to_category(
        df,
        [
            "sex",
            "diagnosis_code",
            "morphology",
            "topography",
            "life_status",
            "surgery_1_intent",
            "surgery_1_margins_after_surgery",
            "surgery_2_intent",
            "surgery_2_margins_after_surgery",
            "surgery_3_intent",
            "surgery_3_margins_after_surgery",
            "surgery_4_intent",
            "surgery_4_margins_after_surgery",
            "surgery_5_intent",
            "surgery_5_margins_after_surgery",
        ],
    )

    df = _to_int64(df, ["year_of_birth", "age_at_diagnosis"])

    df = _to_datetime(
        df,
        [
            "diagnosis_date",
            "life_status_date",
            "surgery_1_date",
            "surgery_2_date",
            "surgery_3_date",
            "surgery_4_date",
            "surgery_5_date",
        ],
    )

    df = _to_boolean(df, 
        [
            "clinical_is_transit_metastasis_with_clinical_confirmation",
            "clinical_is_multifocal_tumor",
            "clinical_regional_nodal_metastases",
            "clinical_soft_tissue",
            "clinical_distant_lymph_node",
            "clinical_lung",
            "clinical_metastasis_at_bone",
            "clinical_liver",
            "clinical_pleura",
            "clinical_peritoneum",
            "clinical_brain",
            "clinical_other_viscera",
            "clinical_unknown",
            "pathological_regional_nodal_metastases",
            "pathological_soft_tissue",
            "pathological_distant_lymph_node",
            "pathological_lung",
            "pathological_metastasis_at_bone",
            "pathological_liver",
            "pathological_pleura",
            "pathological_peritoneum",
            "pathological_brain",
            "pathological_other_viscera",
            "pathological_unknown"
        ]
    )

    return df


def convert_head_neck_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = _to_category(
        df,
        [
            "pathological_stage",
            "clinical_stage",
            "pathological_stage_pt",
            "pathological_stage_pn",
            "pathological_stage_pm",
            "clinical_stage_ct",
            "clinical_stage_cn",
            "clinical_stage_cm",
        ],
    )
    return df

def convert_sarcoma_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _to_category(
        df,
        [
            "type_of_biopsy",
        ],
    )

    df = _to_int64(
        df, 
        [
            "number_of_tumor_nodules"
        ],
    )

    df = _to_datetime(
        df,
        [
            "date_of_biopsy",
            "last_contact",
        ],
    )

    df = _to_boolean(
        df,
        [
            "clinical_localised",
            "clinical_loco_regional",
        ],
    )
    return df



