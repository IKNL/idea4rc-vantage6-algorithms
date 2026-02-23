----------------------------------------------------------------------------------------
-- Head and Neck features
-- This query is used to extract the features from the Head and Neck dataset.

-- VERSION  | AUTHOR         | DATE          | DESCRIPTION
-- -------------------------------------------------------------------------------------
-- 1.0      | F. Martin      | 2026-02-10    | Initial version
-- -------------------------------------------------------------------------------------
-- 1.1      | F. Martin      | 2026-02-11    | Update the clinical staging codes and 
--          | A. van Gestel  |               | fixed some minor issues.
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
-- of episode_concept_id of 32533 Disease Episode (always one) look up the date
-- 
-- Topography and morphology
-- Extract episode_object_concept_id this contains the omop concept id for the ICDO3 
-- code (for topography and morphology).
----------------------------------------------------------------------------------------

    -- EPISODE table
    primary_tumor AS (
        SELECT
            pid.person_id,
            episode.episode_start_date as diagnosis_date, -- e.g. 2020-01-01
            split_part(diagnosis_concept.concept_name, '-', 1) as morphology, -- e.g. 8888/8
            split_part(diagnosis_concept.concept_name, '-', 2) as topography -- e.g. C34.1
        FROM
            patient_id_list pid
        LEFT JOIN
            @cdm_schema.episode episode
            ON pid.person_id = episode.person_id
        LEFT JOIN
            @cdm_schema.concept diagnosis_concept
            ON episode.episode_object_concept_id = diagnosis_concept.concept_id
        WHERE
            episode.episode_concept_id = 32533 -- Disease Episode
    ),

----------------------------------------------------------------------------------------
-- Observation table
--
-- Concept_id look for codes of Alive Dead etc (see excel). Look for latest date. 
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
                    2000100071,     -- Alive, No Evidence of Disease (NED) 
                    4230556,        -- Alive 
                    2000100072,     -- Dead of Disease (DOD) 
                    2000100073,     -- Dead of Other Cause (DOC) 
                    2000100074,     -- Dead of Unknown Cause (DUC) 
                    2000100075,     -- Alive With Disease (AWD) 
                    4163894         -- Lost to follow-up 
                )
        ) AS all_observations
        WHERE all_observations.observation_position = 1 -- get last observation
    ),

----------------------------------------------------------------------------------------
-- Measurement table
--
-- 
-- Pathological staging
-- Measurement_concept_id look up the code from pathological staging
-- 
-- Clinical staging
-- Measurement_concept_id look up the code from clinical staging
-- 
----------------------------------------------------------------------------------------

    -- MEASUREMENT table
    pathological_staging AS (
        SELECT
            pid.person_id,
            measurement_concept.concept_name as stage
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
                1634551         -- AJCC/UICC 8th pathological Stage 4C
            )
    ),


    clinical_staging AS (
        SELECT
            pid.person_id,
            measurement_concept.concept_name as stage
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
                1635842, -- AJCC/UICC 6th clinical Stage 0
                1633828, -- AJCC/UICC 7th clinical Stage 0
                1635824, -- AJCC/UICC 8th clinical Stage 0

                1633905, -- AJCC/UICC 6th clinical Stage 1
                1634457, -- AJCC/UICC 7th clinical Stage 1
                1635758, -- AJCC/UICC 8th clinical Stage 1

                1634718, -- AJCC/UICC 6th clinical Stage 2
                1635182, -- AJCC/UICC 7th clinical Stage 2
                1635217, -- AJCC/UICC 8th clinical Stage 2

                1635848, -- AJCC/UICC 6th clinical Stage 3
                1635125, -- AJCC/UICC 7th clinical Stage 3
                1634596, -- AJCC/UICC 8th clinical Stage 3

                1634307, -- AJCC/UICC 6th clinical Stage 4
                1634766, -- AJCC/UICC 7th clinical Stage 4
                1635029, -- AJCC/UICC 8th clinical Stage 4

                1635535, -- AJCC/UICC 6th clinical Stage 4A
                1634451, -- AJCC/UICC 7th clinical Stage 4A
                1634810, -- AJCC/UICC 8th clinical Stage 4A

                1633922, -- AJCC/UICC 6th clinical Stage 4B
                1635757, -- AJCC/UICC 7th clinical Stage 4B
                1635708, -- AJCC/UICC 8th clinical Stage 4B

                1633270, -- AJCC/UICC 6th clinical Stage 4C
                1634614, -- AJCC/UICC 7th clinical Stage 4C
                1635006  -- AJCC/UICC 8th clinical Stage 4C
            )
    )

----------------------------------------------------------------------------------------
-- Construct final table
----------------------------------------------------------------------------------------
SELECT

    person.person_id                                as patient_id,
    COALESCE(person.sex, 'N/A')                     as sex,
    person.year_of_birth                            as year_of_birth,
    primary_tumor.diagnosis_date                    as diagnosis_date,
    COALESCE(primary_tumor.morphology, 'N/A')       as morphology,
    COALESCE(primary_tumor.topography, 'N/A')       as topography,
    COALESCE(person_status.status, 'N/A')           as life_status,
    person_status.date                              as life_status_date,
    COALESCE(pathological_staging.stage, 'N/A')     as pathological_stage,
    COALESCE(clinical_staging.stage, 'N/A')         as clinical_stage

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



