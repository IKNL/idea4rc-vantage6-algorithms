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


COHORT_R_DATE_COLUMNS = [
    "date_of_surgery",
    "diagnosis_date",
    "life_status_date",
    "date_of_biopsy",
    "last_contact",
    "surgery_1_date",
    "surgery_2_date",
    "surgery_3_date",
    "surgery_4_date",
    "surgery_5_date",
    "pre_operative_systemic_treatment_start_date",
    "pre_operative_systemic_treatment_end_date",
    "post_operative_systemic_treatment_1_start_date",
    "post_operative_systemic_treatment_1_end_date",
    "post_operative_systemic_treatment_2_start_date",
    "post_operative_systemic_treatment_2_end_date",
    "recurrence_systemic_treatment_1_start_date",
    "recurrence_systemic_treatment_1_end_date",
    "recurrence_systemic_treatment_2_start_date",
    "recurrence_systemic_treatment_2_end_date",
]


def _convert_r_date_columns_safe(df: pd.DataFrame, cols: list[str], max_abs_days: float = 120_000) -> pd.DataFrame:
    """R Date is days since 1970-01-01. Outliers overflow vectorized ohdsi/pandas conversion."""
    colmap = {str(c).lower(): c for c in df.columns}
    for want in cols:
        col = colmap.get(want.lower())
        if col is None:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        fv = np.asarray(s, dtype=np.float64)
        finite = np.isfinite(fv)
        plausible = finite & (np.abs(fv) <= max_abs_days)
        # Only convert plausible rows: full-column vectorized unit="D" can overflow
        # if cast_from_unit_vectorized still touches masked outliers (pandas 3.x).
        out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
        pos = np.flatnonzero(plausible)
        if pos.size:
            parsed = pd.to_datetime(
                fv[plausible],
                unit="D",
                origin="1970-01-01",
                utc=True,
                errors="coerce",
            )
            chunk = pd.Series(parsed, dtype="datetime64[ns, UTC]", copy=False)
            out.iloc[pos] = chunk.to_numpy()
        col_idx = list(df.columns).index(col)
        df.drop(columns=[col], inplace=True)
        df.insert(col_idx, col, out)
    return df


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
        converted_df = common.convert_from_r(df, date_cols=[])
        converted_df = _convert_r_date_columns_safe(converted_df, COHORT_R_DATE_COLUMNS)
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
            "pre_operative_systemic_treatment_regimen",
            "pre_operative_systemic_treatment_setting",
            "pre_operative_systemic_treatment_type",
            "pre_operative_systemic_treatment_reason_for_end_of_treatment",
            "post_operative_systemic_treatment_1_regimen",
            "post_operative_systemic_treatment_1_setting",
            "post_operative_systemic_treatment_1_type",
            "post_operative_systemic_treatment_1_reason_for_end_of_treatment",
            "post_operative_systemic_treatment_2_regimen",
            "post_operative_systemic_treatment_2_setting",
            "post_operative_systemic_treatment_2_type",
            "post_operative_systemic_treatment_2_reason_for_end_of_treatment",
            "recurrence_systemic_treatment_1_regimen",
            "recurrence_systemic_treatment_1_setting",
            "recurrence_systemic_treatment_1_type",
            "recurrence_systemic_treatment_1_reason_for_end_of_treatment",
            "recurrence_systemic_treatment_2_regimen",
            "recurrence_systemic_treatment_2_setting",
            "recurrence_systemic_treatment_2_type",
            "recurrence_systemic_treatment_2_reason_for_end_of_treatment",
            "pre_operative_radio_hospital",
            "pre_operative_radio_setting",
            "pre_operative_radio_intent",
            "pre_operative_radio_treatment_completed_as_planned",
            "post_operative_radio_1_hospital",
            "post_operative_radio_1_setting",
            "post_operative_radio_1_intent",
            "post_operative_radio_1_treatment_completed_as_planned",
            "post_operative_radio_2_hospital",
            "post_operative_radio_2_setting",
            "post_operative_radio_2_intent",
            "post_operative_radio_2_treatment_completed_as_planned",
            "recurrence_radio_1_hospital",
            "recurrence_radio_1_setting",
            "recurrence_radio_1_intent",
            "recurrence_radio_1_treatment_completed_as_planned",
            "recurrence_radio_2_hospital",
            "recurrence_radio_2_setting",
            "recurrence_radio_2_intent",
            "recurrence_radio_2_treatment_completed_as_planned",
        ],
    )

    df = _to_int64(
        df, 
        [
            "year_of_birth", 
            "age_at_diagnosis", 
            "pre_operative_radio_total_dose_gy",
            "pre_operative_radio_number_of_fractions",
            "post_operative_radio_1_total_dose_gy",
            "post_operative_radio_1_number_of_fractions",
            "post_operative_radio_2_total_dose_gy",
            "post_operative_radio_2_number_of_fractions",
            "recurrence_radio_1_total_dose_gy",
            "recurrence_radio_1_number_of_fractions",
            "recurrence_radio_2_total_dose_gy",
            "recurrence_radio_2_number_of_fractions",
        ])

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
            "pre_operative_systemic_treatment_start_date",
            "pre_operative_systemic_treatment_end_date",
            "post_operative_systemic_treatment_1_start_date",
            "post_operative_systemic_treatment_1_end_date",
            "post_operative_systemic_treatment_2_start_date",
            "post_operative_systemic_treatment_2_end_date",
            "recurrence_systemic_treatment_1_start_date",
            "recurrence_systemic_treatment_1_end_date",
            "recurrence_systemic_treatment_2_start_date",
            "recurrence_systemic_treatment_2_end_date",
            "pre_operative_radio_start_date",
            "pre_operative_radio_end_date",
            "post_operative_radio_1_start_date",
            "post_operative_radio_1_end_date",
            "post_operative_radio_2_start_date",
            "post_operative_radio_2_end_date",
            "recurrence_radio_1_start_date",
            "recurrence_radio_1_end_date",
            "recurrence_radio_2_start_date",
            "recurrence_radio_2_end_date",
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
            "pathological_unknown",
            "pre_operative_radio_intraoperative_radio",
            "post_operative_radio_1_intraoperative_radio",
            "post_operative_radio_2_intraoperative_radio",
            "recurrence_radio_1_intraoperative_radio",
            "recurrence_radio_2_intraoperative_radio",
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
            "clinical_stage_extra_nodal_extension",
            "pathological_stage_extra_nodal_extension",

            "surgery_1_extra_nodal_extension",
            "surgery_2_extra_nodal_extension",
            "surgery_3_extra_nodal_extension",
            "surgery_4_extra_nodal_extension",
            "surgery_5_extra_nodal_extension",

            "surgery_1_laterality_of_the_dissection",
            "surgery_2_laterality_of_the_dissection",
            "surgery_3_laterality_of_the_dissection",
            "surgery_4_laterality_of_the_dissection",
            "surgery_5_laterality_of_the_dissection",

            "pre_operative_systemic_treatment_intent",
            "post_operative_systemic_treatment_1_intent",
            "post_operative_systemic_treatment_2_intent",
            "recurrence_systemic_treatment_1_intent",
            "recurrence_systemic_treatment_2_intent",

            "pre_operative_radio_beam_quality",
            "post_operative_radio_1_beam_quality",
            "post_operative_radio_2_beam_quality",
            "recurrence_radio_1_beam_quality",
            "recurrence_radio_2_beam_quality",

            "surgery_1_surgery_hospital",
            "surgery_2_surgery_hospital",
            "surgery_3_surgery_hospital",
            "surgery_4_surgery_hospital",
            "surgery_5_surgery_hospital",
        ],
    )
    
    df = _to_int64(
        df, 
        [
            "pre_operative_radio_total_high_dose",
            "post_operative_radio_1_total_high_dose",
            "post_operative_radio_2_total_high_dose",
            "recurrence_radio_1_total_high_dose",
            "recurrence_radio_2_total_high_dose",
        ])
    
    df = _to_datetime(
        df,
        [
            "surgery_1_date_of_neck_surgery",
            "surgery_2_date_of_neck_surgery",
            "surgery_3_date_of_neck_surgery",
            "surgery_4_date_of_neck_surgery",
            "surgery_5_date_of_neck_surgery",
        ],
    )

    df = _to_boolean(df, 
        [
            "surgery_1_neck_surgery",
            "surgery_2_neck_surgery",
            "surgery_3_neck_surgery",
            "surgery_4_neck_surgery",
            "surgery_5_neck_surgery",

            "pre_operative_radio_treatment_site_distant_metastasis",
            "post_operative_radio_1_treatment_site_distant_metastasis",
            "post_operative_radio_2_treatment_site_distant_metastasis",
            "recurrence_radio_1_treatment_site_distant_metastasis",
            "recurrence_radio_2_treatment_site_distant_metastasis"
        ]
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
            "clinical_number_of_tumor_nodules",
            "pathological_number_of_tumor_nodules",
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
            "pathological_localised",
            "pathological_loco_regional",
            "pathological_is_transit_metastasis_with_clinical_confirmation",
            "pathological_is_multifocal_tumor",
        ]
    )
    return df