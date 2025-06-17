WITH
    --- get all patients in the cohort
    person AS (
        SELECT
            cohort.subject_id as person_id,
            gender_concept.concept_name as sex,
            EXTRACT(YEAR FROM cohort.cohort_start_date) - person.year_of_birth as age
        FROM
            omopcdm.cohort cohort
        LEFT JOIN
            omopcdm.person person
            ON cohort.subject_id = person.person_id
        LEFT JOIN
            omopcdm.concept gender_concept
            ON person.gender_concept_id = gender_concept.concept_id
        WHERE cohort_definition_id = 80425
    ),
    --- get all patients in the cohort and their death information
    death AS (
        SELECT
            cohort.subject_id as person_id,
            CAST(CASE WHEN death.death_date IS NOT NULL THEN 1 ELSE 0 END AS BIT) AS censor,
            CASE WHEN death.death_date IS NOT NULL THEN 'DEAD' ELSE 'ALIVE' END AS status,
            COALESCE((death.death_date - cohort.cohort_start_date),(cohort.cohort_end_date - cohort.cohort_start_date)
            ) AS survival_days
        FROM
            omopcdm.cohort cohort
        LEFT JOIN
            omopcdm.death death
            ON cohort.subject_id = death.person_id
        LEFT JOIN
            omopcdm.observation_period op --- questo vale se ho un unico observation_period, altrimenti dovrei prendere l'ultimo (?)
            ON cohort.subject_id = op.person_id
        WHERE cohort_definition_id = 80425
    ),
    --- get tumor grade (if grade after surgery is available, otherwise grade at diagnosis)
    tumor_grade AS (
        SELECT
            all_tumor_grade.subject_id AS person_id,
            all_tumor_grade.grade
        FROM (
            SELECT
                cohort.subject_id,
                measurement.measurement_concept_id,
                grade_concept.concept_name as grade,
                measurement.measurement_date,
                ROW_NUMBER() OVER (PARTITION BY cohort.subject_id ORDER BY measurement.measurement_date DESC) AS rn
            FROM
                omopcdm.cohort cohort
            LEFT JOIN
                omopcdm.measurement measurement
                ON cohort.subject_id = measurement.person_id
            left join
                omopcdm.concept grade_concept
                ON measurement.measurement_concept_id = grade_concept.concept_id
            WHERE
                measurement.measurement_concept_id IN (1634371,1634752,1633749) -- FNCLCC grade
                AND cohort_definition_id = 80425
        ) AS all_tumor_grade
        WHERE rn = 1
    )
    ,
    --- get primary diagnosis for all patients in the cohort (the date is the reference for some of the other variables)
    -- primary_tumor AS (
    --     SELECT
    --         episode.person_id,
    --         episode.episode_id,
    --         episode.episode_concept_id,
    --         episode.episode_start_date as diagnosis_date,
    --         episode.episode_end_date as diagnosis_end_date,
    --         episode.episode_object_concept_id as diagnosis_concept,
    --         diagnosis_concept.concept_name as diagnosis
    --     FROM
    --         omopcdm.cohort cohort
    --     LEFT JOIN
    --         omopcdm.episode episode
    --         ON cohort.subject_id = episode.person_id
    --     LEFT JOIN
    --     	omopcdm.concept diagnosis_concept
    --     	ON episode.episode_object_concept_id = diagnosis_concept.concept_id
    --     WHERE
    --         episode.episode_concept_id = 32533 --- Disease Episode (overarching episode)
    --         AND cohort_definition_id = 80425
    -- ),
    --- get main surgery information
    surgery AS (
        SELECT
            all_surgeries.person_id,
            all_surgeries.episode_start_date as surgery_date,
            po.procedure_concept_id  as surgery_concept
            -- surgery_concept.concept_name as surgery
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY episode.person_id ORDER BY episode.episode_start_date) AS rn
            FROM
                omopcdm.episode episode
            LEFT JOIN
                omopcdm.cohort cohort
                ON cohort.subject_id = episode.person_id
                -- AND episode.episode_start_date = cohort.cohort_start_date
            WHERE
                -- TODO confirm that we have this code in the data
                episode.episode_concept_id = 32939
                -- AND episode.episode_parent_id IN (SELECT primary_tumor.episode_id FROM primary_tumor) --- get the surgeries related only to the overarching episode considered
                -- AND episode.episode_object_concept_id NOT IN (
				-- 	SELECT
                --         c.concept_id
                --     FROM
                --         omopcdm.concept c
                --     JOIN
                --         omopcdm.concept_ancestor ca
                --         ON c.concept_id = ca.descendant_concept_id
                --         AND ca.ancestor_concept_id IN (4311405) -- Biopsy
                --         AND c.invalid_reason IS NULL
				-- 		AND c.domain_id = 'Measurement'
				-- )
                AND cohort_definition_id = 80425
        ) AS all_surgeries
        LEFT join
            omopcdm.episode_event ee
            on all_surgeries.episode_id = ee.episode_id
        LEFT join
            omopcdm.procedure_occurrence po
            on ee.event_id = po.procedure_occurrence_id
        LEFT JOIN
            omopcdm.concept surgery_concept
            ON po.procedure_concept_id = surgery_concept.concept_id
        WHERE rn = 1
    ),
    --- get tumor size (the greater between diagnosis and surgery)
    tumor_size AS (
        SELECT
            all_tumor_size.subject_id AS person_id,
            CASE
                WHEN all_tumor_size.unit_concept_id = 8582 THEN all_tumor_size.value_as_number
                WHEN all_tumor_size.unit_concept_id = 8588 THEN all_tumor_size.value_as_number/10
            END AS tumor_size
        FROM (
            SELECT
                cohort.subject_id,
                measurement.value_as_number,
                measurement.unit_concept_id,
                ROW_NUMBER() OVER (PARTITION BY cohort.subject_id ORDER BY
                                CASE
                                    WHEN measurement.unit_concept_id = 8582 THEN measurement.value_as_number
                                    WHEN measurement.unit_concept_id = 8588 THEN measurement.value_as_number/10
                                END DESC) AS rn
            FROM
                omopcdm.cohort cohort
            LEFT JOIN
                omopcdm.measurement measurement
                ON cohort.subject_id = measurement.person_id
            -- LEFT JOIN
            --     primary_tumor
            --     ON primary_tumor.person_id = measurement.person_id
            --     AND primary_tumor.diagnosis_date = measurement.measurement_date
            LEFT JOIN
                surgery
                ON surgery.person_id = measurement.person_id
                AND surgery.surgery_date = measurement.measurement_date
            WHERE
                measurement.measurement_concept_id IN (36768664,36768255) -- Tumor size concepts
                -- AND (primary_tumor.diagnosis_date IS NOT NULL OR surgery.surgery_date IS NOT NULL)
                AND cohort_definition_id = 80425
        ) AS all_tumor_size
        WHERE rn = 1
    ),
    --- get information about focality of tumor (unifocal or multifocal) at diagnosis
    focality AS (
        SELECT
            cohort.subject_id as person_id,
            measurement.measurement_concept_id,
            upper(focality_concept.concept_name) AS focality
        FROM
            omopcdm.cohort cohort
        LEFT JOIN
            omopcdm.measurement measurement
            ON cohort.subject_id = measurement.person_id
        -- LEFT JOIN
        --     primary_tumor
        --     on primary_tumor.person_id = measurement.person_id
        LEFT JOIN
            omopcdm.concept focality_concept
            ON measurement.measurement_concept_id = focality_concept.concept_id
        WHERE
            measurement.measurement_concept_id IN (36769933,36769332) --- Unifocal Tumor and Multifocal Tumor
            -- AND primary_tumor.diagnosis_date = measurement.measurement_date
            AND cohort_definition_id = 80425
   UNION
    	SELECT
            cohort.subject_id as person_id,
            condition.condition_concept_id,
            upper(focality_concept.concept_name) AS focality
        FROM
            omopcdm.cohort cohort
        LEFT JOIN
            omopcdm.condition_occurrence condition
            ON cohort.subject_id = condition.person_id
        -- LEFT JOIN
        --     primary_tumor
        --     on primary_tumor.person_id = condition.person_id
        LEFT JOIN
            omopcdm.concept focality_concept
            ON condition.condition_concept_id = focality_concept.concept_id
        WHERE
            condition.condition_concept_id IN (4163998,4163442) --- Unifocal tumor and Multifocal tumor
            -- AND primary_tumor.diagnosis_date = condition.condition_start_date
            AND cohort_definition_id = 80425
    ),
    --- get resection information @ main surgery
    resection AS (
        SELECT
            cohort.subject_id as person_id,
            measurement.measurement_concept_id as measurement_concept_id,
            resection_concept.concept_name AS resection,
            -- TODO: IFF is needed for the parameterized version of the query
            --IIF(measurement.measurement_concept_id in (1634643,1633801), 'Macroscopically complete', 'Macroscopically incomplete') AS completeness_of_resection
            CASE
                -- TODO this does not seem to be valid, as now M complete includes small
                -- residual tumor
                WHEN measurement.measurement_concept_id = 1634643 THEN 'Macroscopically complete'
                WHEN measurement.measurement_concept_id = 1633801 THEN 'Macroscopically complete'
                WHEN measurement.measurement_concept_id = 1634484 THEN 'Macroscopically incomplete'
                ELSE 'N/A'
            END AS completeness_of_resection
        FROM
            omopcdm.cohort cohort
        LEFT JOIN
            omopcdm.measurement measurement
            ON cohort.subject_id = measurement.person_id
        left join
            surgery
            on surgery.person_id = measurement.person_id
        LEFT JOIN
            omopcdm.concept resection_concept
            ON measurement.measurement_concept_id = resection_concept.concept_id
        WHERE
            measurement.measurement_concept_id IN (1634643,1633801,1634484) --- R0, R1, R2
            -- TODO: we commented this out, but do we need to have the surgery date in
            -- here?
            -- AND surgery.surgery_date = measurement.measurement_date
            AND cohort_definition_id = 80425
    ),
    --- get tumor rupture after main surgery
    tumor_rupture AS (
        SELECT
            cohort.subject_id as person_id,
            measurement.measurement_concept_id
        FROM
            omopcdm.cohort cohort
        LEFT JOIN
            omopcdm.measurement measurement
            ON cohort.subject_id = measurement.person_id
        left join
            surgery
            on surgery.person_id = measurement.person_id
        WHERE
            measurement.measurement_concept_id = 36768904 --- Tumor Rupture
            -- TODO: we commented this out, but do we need to have the surgery date in
            -- here?
            -- AND surgery.surgery_date = measurement.measurement_date
            AND cohort_definition_id = 80425
    ),
    --- get local recurrence information
    -- TODO concept ID was not specified in the IDEA4RC datamodel.
    recurrence AS (
        SELECT
            all_recurrence.subject_id AS person_id,
            all_recurrence.condition_start_date AS recurrence_date
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY co.person_id ORDER BY co.condition_start_date) AS rn
            FROM
                omopcdm.cohort cohort
            LEFT JOIN
                omopcdm.condition_occurrence co
                ON cohort.subject_id = co.person_id
            -- LEFT JOIN
            --     primary_tumor
            --     ON cohort.subject_id = primary_tumor.person_id
            JOIN
                (SELECT
                    *
                FROM
                    omopcdm.concept c
                JOIN
                    omopcdm.concept_ancestor ca
                    ON c.concept_id = ca.descendant_concept_id
                    AND ca.ancestor_concept_id IN (4097297) --- Recurrent neoplasm
                    AND c.invalid_reason IS NULL
                ) AS recurrence_concept
                ON co.condition_concept_id = recurrence_concept.descendant_concept_id
            WHERE
                -- TODO: the original query uses DATEDIFF(day,primary_tumor.diagnosis_date, co.condition_start_date) > 0
                -- (co.condition_start_date - primary_tumor.diagnosis_date) > 0
                -- AND
                cohort_definition_id = 80425
        ) AS all_recurrence
        WHERE rn = 1
    ),
    --- Pre-operative chemotherapy
    pre_chemo AS (
        SELECT
            all_pre_chemo.subject_id AS person_id,
            all_pre_chemo.episode_start_date as pre_chemo_date
        FROM (
            SELECT
                    *,
                    ROW_NUMBER() OVER (PARTITION BY episode.person_id ORDER BY episode.episode_start_date DESC) AS rn
                FROM
                    omopcdm.episode episode
                LEFT JOIN
                    omopcdm.cohort cohort
                    ON cohort.subject_id = episode.person_id
                -- LEFT JOIN
                --     primary_tumor
                --     ON cohort.subject_id = primary_tumor.person_id
                JOIN
                    surgery
                    ON cohort.subject_id = surgery.person_id
                    -- TODO the '<' might be silly
                    AND episode.episode_start_date <= surgery.surgery_date
                JOIN
                    omopcdm.procedure_occurrence po
                    ON po.person_id = cohort.subject_id
                        AND po.procedure_date = episode.episode_start_date
                        -- AND po.procedure_end_date = episode.episode_end_date
                WHERE
                    episode.episode_concept_id IN (32531,32941) --- Treament regimen or Cancer Drug Treatment
                    -- AND episode.episode_parent_id IN (SELECT primary_tumor.episode_id FROM primary_tumor)
                    AND po.procedure_concept_id IN (
                        SELECT
                            c.concept_id
                        FROM
                            omopcdm.concept c
                        JOIN
                            omopcdm.concept_ancestor ca
                            ON c.concept_id = ca.descendant_concept_id
                            AND ca.ancestor_concept_id IN (4273629) --- Chemotherapy and all descendants
                            AND c.invalid_reason IS NULL
                        )
                    AND cohort_definition_id = 80425
                ) AS all_pre_chemo
            WHERE rn = 1
    ),
    --- Post-operative chemotherapy
    post_chemo AS (
        SELECT
            all_post_chemo.subject_id AS person_id,
            all_post_chemo.episode_start_date AS post_chemo_date
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY episode.person_id ORDER BY episode.episode_start_date) AS rn
            FROM
                omopcdm.episode episode
            LEFT JOIN
                omopcdm.cohort cohort
                ON cohort.subject_id = episode.person_id
            -- LEFT JOIN
            --     primary_tumor
            --     ON cohort.subject_id = primary_tumor.person_id
            JOIN
                surgery
                ON cohort.subject_id = surgery.person_id
                    AND episode.episode_start_date > surgery.surgery_date
            LEFT JOIN
                recurrence
                ON cohort.subject_id = recurrence.person_id
            JOIN
                omopcdm.procedure_occurrence po
                ON po.person_id = cohort.subject_id
                    AND po.procedure_date = episode.episode_start_date
                    -- AND po.procedure_end_date = episode.episode_end_date
            WHERE
                episode.episode_concept_id IN (32531,32941) --- Treament regimen or Cancer Drug Treatment
                -- AND episode.episode_parent_id IN (SELECT primary_tumor.episode_id FROM primary_tumor)
                -- AND episode.episode_start_date < ISNULL(recurrence.recurrence_date, primary_tumor.diagnosis_end_date)
                -- AND episode.episode_start_date < COALESCE(recurrence.recurrence_date, primary_tumor.diagnosis_end_date)
                AND po.procedure_concept_id IN (
                    SELECT
                        c.concept_id
                    FROM
                        omopcdm.concept c
                    JOIN
                        omopcdm.concept_ancestor ca
                        ON c.concept_id = ca.descendant_concept_id
                        AND ca.ancestor_concept_id IN (4273629) --- Chemotherapy and all descendants
                        AND c.invalid_reason IS NULL)
                AND cohort_definition_id = 80425
            ) AS all_post_chemo
        WHERE rn = 1
    ),
    --- Pre-operative radiotherapy
    -- TODO how to compute this from the IDEA4RC datamodel?
    pre_radio AS (
        SELECT
            all_pre_radio.subject_id AS person_id,
            all_pre_radio.episode_start_date AS pre_radio_date
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY episode.person_id ORDER BY episode.episode_start_date DESC) AS rn
            FROM
                omopcdm.episode episode
            LEFT JOIN
                omopcdm.cohort cohort
                ON cohort.subject_id = episode.person_id
            -- LEFT JOIN
            --     primary_tumor
            --     ON cohort.subject_id = primary_tumor.person_id
            JOIN
                surgery
                on cohort.subject_id = surgery.person_id AND episode.episode_start_date < surgery.surgery_date
            WHERE
                episode.episode_concept_id = 32940
                -- AND episode.episode_parent_id IN (SELECT primary_tumor.episode_id FROM primary_tumor)
                AND cohort_definition_id = 80425
            ) AS all_pre_radio
        WHERE rn = 1
    ),
    --- Post-operative radiotherapy
    post_radio AS (
        SELECT
            all_post_radio.subject_id AS person_id,
            all_post_radio.episode_start_date as post_radio_date
        FROM (
            SELECT
                    *,
                    ROW_NUMBER() OVER (PARTITION BY episode.person_id ORDER BY episode.episode_start_date) AS rn
                FROM
                    omopcdm.episode episode
                LEFT JOIN
                    omopcdm.cohort cohort
                    ON cohort.subject_id = episode.person_id
                -- LEFT JOIN
                --     primary_tumor
                --     ON cohort.subject_id = primary_tumor.person_id
                JOIN
                    surgery
                    ON cohort.subject_id = surgery.person_id AND episode.episode_start_date > surgery.surgery_date
                LEFT JOIN
                    recurrence
                    on cohort.subject_id = recurrence.person_id
                WHERE
                    episode.episode_concept_id = 32940
                    -- AND episode.episode_parent_id IN (SELECT primary_tumor.episode_id FROM primary_tumor) --- get the radiotherapies related only to the overarching episode considered
                    -- AND episode.episode_start_date < ISNULL(recurrence.recurrence_date, primary_tumor.diagnosis_end_date)
                    -- AND episode.episode_start_date < COALESCE(recurrence.recurrence_date, primary_tumor.diagnosis_end_date)
                    AND cohort_definition_id = 80425
        ) AS all_post_radio
        WHERE rn = 1
    ),
    --- get distant metastasis information
    -- TODO in the IDEA4RC datamodel they use a different code. We need to check
    -- what is the way togo.
    metastasis AS (
        SELECT
            cohort.subject_id as person_id,
            count(*) as n_metastasis
        FROM
            omopcdm.cohort cohort
        LEFT JOIN
            omopcdm.measurement m
            ON cohort.subject_id = m.person_id
        -- LEFT JOIN
        --     primary_tumor
        --     ON cohort.subject_id = primary_tumor.person_id
        -- JOIN
        --     (SELECT
        --         *
        --     FROM
        --         omopcdm.concept c
        --     JOIN
            --     omopcdm.concept_ancestor ca
            --     ON c.concept_id = ca.descendant_concept_id
            --     -- TODO: do we only need to check for regional spread to lymph nodes.
            --     --       36769269
            --     AND ca.ancestor_concept_id IN (36769180) --- Metastasis
            --     AND c.invalid_reason IS NULL
            -- ) AS metastasis_concept
                -- ON m.measurement_concept_id = 36769269 --- Regional spread to lymph nodes
        WHERE
            -- TODO: the original query uses DATEDIFF(day,primary_tumor.diagnosis_date, m.measurement_date) > 90
            -- (m.measurement_date - primary_tumor.diagnosis_date) > 90
            -- AND
            m.measurement_concept_id = 36769269
            AND cohort_definition_id = 80425
        GROUP BY
            cohort.subject_id
    ),
    --- get count of cancer episodes for every patient
    n_cancer_episodes_per_patient AS (
        SELECT
            cohort.subject_id as person_id,
            count(*) as n_cancer_episodes
        FROM
            omopcdm.cohort cohort
        LEFT JOIN
            omopcdm.episode episode
            ON cohort.subject_id = episode.person_id
        WHERE
            -- TODO: We need to validate that this is the correct concept_id to select
            -- on.
            episode.episode_concept_id = 32533 --- Disease Episode (overarching episode)
            AND cohort_definition_id = 80425
        GROUP BY
            cohort.subject_id
    )
SELECT
    person.person_id as Patient_ID,
    person.age as Age,
    COALESCE(person.sex,'N/A') as Sex,
    death.censor as Censor,
    death.status as Status,
    death.survival_days as Survival_days,
    COALESCE(tumor_grade.grade,'N/A') as FNCLCC_grade,
    tumor_size.tumor_size as Tumor_size,
    -- primary_tumor.diagnosis as Primary_tumor_diagnosis,
    surgery.surgery_date as Surgery_date,
    surgery.surgery_concept as Surgery_concept,
    COALESCE(focality.focality,'N/A') as Multifocality,
    COALESCE(resection.completeness_of_resection,'N/A2') as Completeness_of_resection,
    resection.measurement_concept_id as Completeness_of_resection_concept_id,
    -- TODO: IFF is needed for the parameterized version of the query
    --CAST(IIF(tumor_rupture.measurement_concept_id IS NOT NULL, 1, 0) AS BIT) AS Tumor_rupture
    CASE
        WHEN tumor_rupture.measurement_concept_id IS NOT NULL THEN 1
        ELSE 0
    END AS Tumor_rupture,
    -- CAST(IIF(pre_chemo.pre_chemo_date IS NOT NULL, 1, 0) AS BIT) as Pre_operative_chemo,
    -- CAST(IIF(post_chemo.post_chemo_date IS NOT NULL, 1, 0) AS BIT) as Post_operative_chemo,
    CASE
        WHEN pre_chemo.pre_chemo_date IS NOT NULL THEN 1
        ELSE 0
    END AS Pre_operative_chemo,
    CASE
        WHEN post_chemo.post_chemo_date IS NOT NULL THEN 1
        ELSE 0
    END AS Post_operative_chemo,
    -- CAST(IIF(pre_radio.pre_radio_date IS NOT NULL, 1, 0) AS BIT) as Pre_operative_radio,
    -- CAST(IIF(post_radio.post_radio_date IS NOT NULL, 1, 0) AS BIT) as Post_operative_radio
    CASE
        WHEN pre_radio.pre_radio_date IS NOT NULL THEN 1
        ELSE 0
    END AS Pre_operative_radio,
    CASE
        WHEN post_radio.post_radio_date IS NOT NULL THEN 1
        ELSE 0
    END AS Post_operative_radio,
    -- CAST(IIF(recurrence.recurrence_date IS NOT NULL, 1, 0) AS BIT) as local_recurrence,
    CASE
        WHEN recurrence.recurrence_date IS NOT NULL THEN 1
        ELSE 0
    END AS Local_recurrence,
    -- CAST(IIF(metastasis.n_metastasis > 0, 1, 0) AS BIT) as distant_metastasis,
    CASE
        WHEN metastasis.n_metastasis > 0 THEN 1
        ELSE 0
    END AS Distant_metastasis,
    COALESCE(n_cancer_episodes_per_patient.n_cancer_episodes, 0) as n_cancer_episodes
FROM
    person
LEFT JOIN
    death
    ON person.person_id = death.person_id
LEFT JOIN
    tumor_grade
    ON person.person_id = tumor_grade.person_id
LEFT JOIN
    tumor_size
    ON person.person_id = tumor_size.person_id
LEFT JOIN
    focality
    ON person.person_id = focality.person_id
LEFT JOIN
    resection
    ON person.person_id = resection.person_id
LEFT JOIN
    tumor_rupture
    ON person.person_id = tumor_rupture.person_id
LEFT JOIN
    pre_chemo
    ON person.person_id = pre_chemo.person_id
LEFT JOIN
    post_chemo
    ON person.person_id = post_chemo.person_id
LEFT JOIN
    pre_radio
    ON person.person_id = pre_radio.person_id
LEFT JOIN
    post_radio
    ON person.person_id = post_radio.person_id
LEFT JOIN
    recurrence
    ON person.person_id = recurrence.person_id
LEFT JOIN
    metastasis
    ON person.person_id = metastasis.person_id
LEFT JOIN
    n_cancer_episodes_per_patient
    ON person.person_id = n_cancer_episodes_per_patient.person_id
LEFT JOIN
    surgery
    ON person.person_id = surgery.person_id