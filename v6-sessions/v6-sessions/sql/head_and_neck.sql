----------------------------------------------------------------------------------------
-- Head and Neck features
-- This query is used to extract the features from the Head and Neck dataset.

-- VERSION  | AUTHOR        | DATE          | DESCRIPTION
-- -------------------------------------------------------------------------------------
-- 1.0      | Frank Martin  | 2026-02-10    | Initial version
-- -------------------------------------------------------------------------------------

WITH

    -- Define the list of patient IDs to analyze
    patient_id_list AS (
        SELECT person_id FROM (VALUES {@patient_ids}) AS t(person_id)
    ),

----------------------------------------------------------------------------------------
-- PERSON table
--
-- Sex
-- In `PERSON.gender_concept_id` there are two codes:
-- 1. Female: 8507
-- 2. Male: 8532
-- This results in a categorical variable with two levels: "Female" and "Male".
--
-- Year of Birth
-- In `PERSON.year_of_birth` there is the year of birth of the patient.
-- This results in a Integer variable with the year of birth.
----------------------------------------------------------------------------------------

    -- PERSON table
    person AS (
        SELECT
            pid.person_id, -- e.g. 12345
            gender_concept.concept_name as sex, -- e.g. "Female" or "Male"
            person.year_of_birth -- e.g. 1990
        FROM
            patient_id_list pid
        LEFT JOIN
            @cdm_schema.person person 
            ON pid.person_id = person.person_id
        LEFT JOIN
            @cdm_schema.concept gender_concept 
            ON person.gender_concept_id = gender_concept.concept_id
    ),

----------------------------------------------------------------------------------------
-- EPISODE table
--
-- Diagnosis date
-- of episode_concept_id of 32533 (always one) look up the date
-- 
-- Morphology and histology
-- Extract episode_object_concept_id this contains the omop concept id for the ICDO3 
-- code (for morph and hist). Split later (in Python?) into morphology and histology.
----------------------------------------------------------------------------------------

    -- EPISODE table
    primary_tumor AS (
        SELECT
            pid.person_id,
            episode.episode_start_date as diagnosis_date, -- e.g. 2020-01-01
            diagnosis_concept.concept_name as diagnosis_concept -- e.g. "???? (split))"
        FROM
            patient_id_list pid
        LEFT JOIN
            @cdm_schema.episode episode
            ON pid.person_id = episode.person_id
        LEFT JOIN
            @cdm_schema.concept diagnosis_concept
            ON episode.episode_object_concept_id = diagnosis_concept.concept_id
        WHERE
            episode.episode_concept_id = 32533
    ),

----------------------------------------------------------------------------------------
-- Observation table
--
-- Concept_id look for codes of Alive Dead etc (see excel see). Look for latest date. 
-- Translate from the vocabulary (also store the date)
----------------------------------------------------------------------------------------

    -- OBSERVATION table
    person_status AS (

        SELECT 
            person_id,
            status,
            observation_datetime as date
        FROM (
            SELECT 
                pid.person_id,
                observation_concept.concept_name as status,
                observation.observation_datetime,
                ROW_NUMBER() OVER (
                    PARTITION BY pid.person_id
                    ORDER BY observation.observation_datetime DESC
                ) AS observation_position
            FROM
                patient_id_list pid
            LEFT JOIN
                @cdm_schema.observation observation
                ON pid.person_id = observation.person_id
            LEFT JOIN
                @cdm_schema.concept observation_concept
                ON observation.observation_concept_id = observation_concept.concept_id
            WHERE
                observation.observation_concept_id IN ( 	
                    2000100071,     -- Alive, No Evidence of Disease (NED) - 4230556
                    4230556,        -- Alive - 
                    2000100072,     -- Dead of Disease (DOD) - 
                    2000100073,     -- Dead of Other Cause (DOC) - 
                    2000100074,     -- Dead of Unknown Cause (DUC) - 
                    2000100075,     -- Alive With Disease (AWD) - 
                    4163894,        -- Lost to follow-up - 
                )
        ) AS all_observations
        WHERE all_observations.observation_position = 1 -- get last observation
    ),

----------------------------------------------------------------------------------------
-- Measurement table
--
-- measurement_concept_id look up the code from Clinical Staging (see excel) in the 
-- Pathological and Clinical tabs.
----------------------------------------------------------------------------------------

    -- MEASUREMENT table
    pathological_staging AS (
        SELECT
            pid.person_id,
            measurement_concept.concept_name as stage,
        FROM 
            patient_id_list pid
        LEFT JOIN
            @cdm_schema.measurement measurement
            ON pid.person_id = measurement.person_id
        LEFT JOIN
            @cdm_schema.concept measurement_concept
            ON measurement.measurement_concept_id = measurement_concept.concept_id
        WHERE
            measurement.measurement_concept_id IN (
                -- Pathological
                1634741,        -- AJCC/UICC 6th pathological Stage 0
                1635511,        -- AJCC/UICC 7th pathological Stage 0
                1634787,        -- AJCC/UICC 8th pathological Stage 0

                1635797,        -- AJCC/UICC 6th pathological Stage 1
                1635800,        -- AJCC/UICC 7th pathological Stage 1
                1634799,        -- AJCC/UICC 8th pathological Stage 1

                1633751,        -- AJCC/UICC 6th pathological Stage 2
                1634619,        -- AJCC/UICC 7th pathological Stage 2
                1635386,        -- AJCC/UICC 8th pathological Stage 2

                1633499,        -- AJCC/UICC 6th pathological Stage 3
                1634947,        -- AJCC/UICC 7th pathological Stage 3
                1634705,        -- AJCC/UICC 8th pathological Stage 3

                1634208,        -- AJCC/UICC 6th pathological Stage 4
                1635230,        -- AJCC/UICC 7th pathological Stage 4
                1633697,        -- AJCC/UICC 8th pathological Stage 4

                1634731,        -- AJCC/UICC 6th pathological Stage 4A
                1635745,        -- AJCC/UICC 7th pathological Stage 4A
                1634005,        -- AJCC/UICC 8th pathological Stage 4A

                1635370,        -- AJCC/UICC 6th pathological Stage 4B
                1634472,        -- AJCC/UICC 7th pathological Stage 4B
                1634487,        -- AJCC/UICC 8th pathological Stage 4B

                1635893,        -- AJCC/UICC 6th pathological Stage 4C
                1634492,        -- AJCC/UICC 7th pathological Stage 4C
                1634551,        -- AJCC/UICC 8th pathological Stage 4C
            )
    ),


    clinical_staging AS (
        SELECT
            pid.person_id,
            measurement_concept.concept_name as stage,
        FROM 
            patient_id_list pid
        LEFT JOIN
            @cdm_schema.measurement measurement
            ON pid.person_id = measurement.person_id
        LEFT JOIN
            @cdm_schema.concept measurement_concept
            ON measurement.measurement_concept_id = measurement_concept.concept_id
        WHERE
            measurement.measurement_concept_id IN (
                -- Clinical
                1635104,        -- AJCC/UICC 6th clinical NX Category
                1633679,        -- AJCC/UICC 7th clinical NX Category
                1634797,        -- AJCC/UICC 8th clinical NX Category

                1633315,        -- AJCC/UICC 6th clinical N0 Category
                1633942,        -- AJCC/UICC 7th clinical N0 Category
                1634070,        -- AJCC/UICC 8th clinical N0 Category

                1635697,        -- AJCC/UICC 6th clinical N1 Category
                1634139,        -- AJCC/UICC 7th clinical N1 Category
                1633651,        -- AJCC/UICC 8th clinical N1 Category

                1635470,        -- AJCC/UICC 6th clinical N2 Category
                1635634,        -- AJCC/UICC 7th clinical N2 Category
                1633763,        -- AJCC/UICC 8th clinical N2 Category

                1634143,        -- AJCC/UICC 6th clinical N2a Category
                1635739,        -- AJCC/UICC 7th clinical N2a Category
                1633788,        -- AJCC/UICC 8th clinical N2a Category

                1633433,        -- AJCC/UICC 6th clinical N2b Category
                1635677,        -- AJCC/UICC 7th clinical N2b Category
                1633323,        -- AJCC/UICC 8th clinical N2b Category

                1634678,        -- AJCC/UICC 6th clinical N2c Category
                1634727,        -- AJCC/UICC 7th clinical N2c Category
                1633271,        -- AJCC/UICC 8th clinical N2c Category

                1635605,        -- AJCC/UICC 6th clinical N3 Category
                1634037,        -- AJCC/UICC 7th clinical N3 Category
                1633854,        -- AJCC/UICC 8th clinical N3 Category

                1633434,        -- AJCC/UICC 6th clinical N3a Category
                1635004,        -- AJCC/UICC 7th clinical N3a Category
                1635496,        -- AJCC/UICC 8th clinical N3a Category

                1635283,        -- AJCC/UICC 6th clinical N3b Category
                1635084,        -- AJCC/UICC 7th clinical N3b Category
                1635828,        -- AJCC/UICC 8th clinical N3b Category
            )
    )

----------------------------------------------------------------------------------------
-- Construct final table
--
--
----------------------------------------------------------------------------------------
SELECT

    person.sex                      as sex,
    person.year_of_birth            as year_of_birth,
    primary_tumor.diagnosis_date    as diagnosis_date,
    primary_tumor.diagnosis_concept as morph_and_topo, -- to be split
    person_status.status            as life_status,
    person_status.date              as life_status_date,
    pathological_staging.stage      as pathological_stage,
    clinical_staging.stage          as clinical_stage

FROM
    person
LEFT JOIN
    primary_tumor
    ON person.person_id = primary_tumor.person_id
LEFT JOIN
    person_status
    ON person.person_id = person_status.person_id
LEFT JOIN
    pathological_staging
    ON person.person_id = pathological_staging.person_id
LEFT JOIN
    clinical_staging
    ON person.person_id = clinical_staging.person_id



