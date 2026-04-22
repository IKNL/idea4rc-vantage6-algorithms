<img src="https://github.com/IKNL/guidelines/blob/master/resources/logos/iknl_nl.png?raw=true" width=200 align="right">

# IDEA4RC Deliverables

This folder contains all documentation required for the vantage6 components to be integrated into the IDEA4RC ecosystem.

**Contents**
- [Introduction](#introduction)
- [User Workflow](#user-workflow)
- [Task Creation](#task-creation)
- [Data Types](#data-types)
- [Data Extraction](#data-Extraction)
  - [Head and Neck](#head-and-neck-variables)
  - [Sarcoma](#sarcoma-variables)
- [Authentication](#authentication)
- [Analytics Algorithms](#analytics-algorithms)
  - [Summary & Table1](#summary-and-table1)
  - [Crosstabs & Chi-Square (Contingency table)](#crosstabs--chi-square-contingency-table)
  - [T-test](#t-test)
  - [Kaplan Meier and Log Rank](#kaplan-meier-and-log-rank)
  - [GLM](#glm)
- [Preprocessing Algorithms](#preprocessing-algorithms)
  - [Timedelta](#timedelta)
  - [Merge categories](#merge-categories)
  - [One hot encoding](#one-hot-encoding)
  - [Merge variables](#merge-variables)
  

## Introduction
This folder collects all documentation and files required to support the integration and validation of vantage6 components in the IDEA4RC project. Here you will find materials and guides for algorithm testing, integrating with the RAVEN UI, security and privacy reviews, and deployment configuration for orchestrator and capsule environments. Use this folder as the starting point for understanding how vantage6 fits within the IDEA4RC infrastructure and to access all technical resources necessary for local and federated analytics development, testing, and deployment.

**Folder structure**

```bash
deliverables/
├── algorithm-tests/ # Local testing of the algorithms used by IKNL team
│   ├── data/ # local test data
│   │   ├── ETL_from_omop.ipynb
│   │   ├── pelvis_cohort.parquet
│   │   ├── rps_cohort.parquet
│   │   └── rps_pelvis_cohort.parquet
│   ├── crosstabs-and-chisq.ipynb
│   ├── analysis-coxph.ipynb
│   ├── glm.ipynb
│   ├── kaplan-meier.ipynb
│   ├── preprocessing_merge_categories.ipynb
│   ├── preprocessing_merge_variables.ipynb
│   ├── preprocessing_one_hot_encoding.ipynb
│   ├── preprocessing_timedelta.ipynb
│   ├── summary.ipynb
│   └── t-test.ipynb
├── raven-api-documentation/ # Documentation for integrating vantage6 into RAVEN
│   ├── 0-new-workspace.ipynb
│   ├── 1-new-analysis.ipynb
│   ├── 2-new-cohort.ipynb
│   ├── 3-data-preparation.ipynb
│   ├── 4-analytics-crosstabs-and-chisquared.ipynb
│   ├── 6-analytics-t-test.ipynb
│   ├── 7-analytics-kaplan-meier-and-log-rank.ipynb
│   ├── 8-analytics-glm.ipynb
│   ├── 9-preprocessing-time-delta.ipynb
│   ├── 10-preprocessing-merge-categories.ipynb
│   ├── 11-preprocessing-one-hot-encoding.ipynb
│   ├── 12-preprocessing-merge-variables.ipynb
│   ├── 13-preprocessing-drop-variable.ipynb
│   ├── 14-preprocessing-basic-arithmetic.ipynb
│   └── token.txt # used for authentication in the 0-X notebooks, not in the repo. Create yourself.
├── security-and-privacy/ # Security analysis per algorithm required by the CoEs
│   ├── Security & Privacy Summary.pdf      
│   ├── Security & Privacy Crosstab.pdf     
│   ├── Security & Privacy GLM.pdf
│   ├── Security & Privacy Kaplan-Meier.pdf
│   └── Security & Privacy t-test.pdf
├── vantage6-configuration/ # Configurations of the deployment
│   └── orchestrator-deployment/ # Helm files used for the deployment at the orchestrator
│       ├── keycloak-values.yaml     # Example Helm values for server deployment
│       ├── server-values.yaml       # Example Helm values for server deployment
│       └── store-values.yaml        # Example Helm values for capsule deployment
└── README.md
```

## Vantage6 Architecture for IDEA4RC
Several changes have been made in the vantage6 infrastucture at core level to support the IDEA4RC usecases:

- The vantage6 node no longer uses the `docker.sock` but connect to the Kubernetes API. This change does not change anything from the users perpective but has great impact on the deployment. In IDEA4RC the capsules architecture is based on Kubernetes which did not support exposing the `docker.sock` to the containers. To see all the changes that have been made, see [this](https://github.com/vantage6/vantage6/issues/248) issue. The same process has been performed for the orchestrator components. The Helm configuration files that have been used in the UPM orchestrator and in the capsules can be found in the [/vantage6-configuration](./vantage6-configuration/orchestrator-deployment/) folder.
- The data extraction job is seperated from the analytics. This had multiple advantages: 
    - Extracting data from an OMOP source, especially when complex querries are needed, is expensive. Doing this for every algorithm and for every step in the algorithm is a waste of time. This was especially important for iterative algorithms like the GLM and CoxPH. 
    - If you have a tabular data frame (the result of the data extraction job) before you do your analytics you can profide the user with more guidance as you have the variables (column names), their types and possibly some other metadata.
    - Algorithms are no longer specific to a specific database type as all analytics expect a tabular data frame which has been profided by the data extraction job
  
  
```mermaid

flowchart LR

subgraph vantage6

  extraction
  parquet@{shape: win-pane}
  analytics@{shape: procs}
  preprocessing

end

subgraph DATABASES 
  direction TB
  FHIR@{shape: cyl, label: "FHIR DB"}
  OMOP@{shape: cyl, label: "OMOP DB"}
  IDEA4RC@{shape: cyl, label: "IDEA4RC DB"}
end


FHIR <--> IDEA4RC
IDEA4RC <--> OMOP


OMOP --> extraction
extraction --> parquet
parquet --> preprocessing
preprocessing --> parquet
parquet --> analytics

parquet ~~~ A@{ shape: comment, label: "The parquet files are stored in the vantage6 session."}
```



## User Workflow
The workflow from a vantage6 point of view is as follows:

```mermaid
flowchart LR

classDef task stroke:#f00

start@{ shape: sm-circ, label: "Start" }
stop@{ shape: framed-circle, label: "Stop" }
authentication@{label: "Authenticate"}
study@{label: "Create Study\n*Raven:new workspace*"}
session@{label: "Create Session\n*Raven:new analysis*"}
cohort@{label: "Create cohort(s)"}
summary@{label: "Data preparation"}
table1@{label: "Table1"}
km@{label: "Kaplan Meier & Log Rank"}
glm@{label: "GLM"}
crosstabs@{label: "Crosstabs & Chi-Squared"}
t-test@{label: "T-test"}
other@{label: "Other analytics"}
variable@{label: "New variable"}

class variable,summary,cohort,crosstabs,glm,t-test,other,table1,km task

start --> authentication
authentication --> Preparation
subgraph Preparation
 study -- "can have multiple" --> session
 session -- "can have multiple" --> cohort
end
authentication --> summary
Preparation --> summary

summary --> variable
variable --> summary

subgraph Analytics
    table1
    km
    glm
    crosstabs
    t-test
    other
end
summary --> table1 
summary --> km
summary --> glm
summary --> crosstabs
summary --> t-test
summary --> other

Analytics --> stop
```

The connection to RAVEN and the database is as follows:

* For a new workspace (RAVEN) we create a new study in vantage6.
* For a new analysis (RAVEN) we create a new session in vantage6.
* For a new cohort (RAVEN) we create a new dataframe in the vantage6 session.

So each session can contain multiple dataframes (cohorts). Each cohort is either a sarcoma or head and neck cohort. This determines which variables are extracted from the database. All cohorts in each (RAVEN) analysis have in this instance the same variables. This is important because vantage6 algorithms can run on multiple cohorts at the same time. This obviously works only when these cohorts have the same data available. See the section on [data extraction](#data-extraction) for more details on the extraction process.

Users in RAVEN are also allowed to create new variables. Since we need to keep the dataframes (cohorts) in sync in terms of variables (columns) we need to apply every preprocessing step to all dataframes so to keep the data aligned.

>[!NOTE]
> In case a preprocessing step fails, vantage6 wont allow any analytics to be run because it can not verify that all nodes have the same data at that point. A dataframe in vantage6 is a *federated* dataframe, thus modifying a single dataframe in practice means that we need to visit all data stations to apply it. In case one of the centers fails, the dataframe becomes out of sync between the centers. To avoid that vantage6 blocks the dataframe, all preprocessing steps will pass (using `try-except` style catching). However that will put pressure on us to write preprocessing algorithms that never fail.

In order to limit the possibilites of preprocessing and analytics we only allow certain data types, see [data-types](#data-types). This makes the environment way more controlled.



## Task creation 
In the flow described above all the `Create cohorts`, `Data preperation`, `Analysis` and `New variable` require a vantage6 task. To create these a single vantage6 endpoint is used in which the payload of the POST request differs for each tasks. The exact payload is given in the notebooks, for example the [data preparation](raven-api-documentation/3-data-preparation.ipynb). The flow is however always the same:

```mermaid
flowchart LR

start@{ shape: sm-circ, label: "Start" }
stop@{ shape: framed-circle, label: "Stop" }
payload@{ shape: lean-r, label: "Analytics payload"}
authorization@{ shape: lean-r, label: "Authorization Header"}
task@{label: "POST task\n**payload**"}
status@{label: "GET status\n*poll if ready*"}
results@{label: "GET results"}

payload --> task
authorization --> task
task --> status
status --> status
status -- "when ready" --> results
start --> task
results --> stop

```


## Data types 
Vantage6 loads the data from the OMOP database. When it does so, it assigns specific numpy/pandas types to all extracted variables. This helps us: 

* In the user interface when the user is asked to select variables of a certain type. For example in Kaplan Meier analysis the user needs to select the survival time variable which requires to be a positive int. 
* In the algorithm when we need to convert one type in another. For example when we need a boolean to be represented as a `0` and `1` instead of `false` and `true`.

We accept the following (numpy) types:

* [Int64Dtype (pandas)](https://pandas.pydata.org/docs/user_guide/integer_na.html)
* [Float64Dtype (pandas)](https://pandas.pydata.org/docs/reference/api/pandas.Float64Dtype.html#pandas.Float64Dtype)
* [boolean (pandas)](https://pandas.pydata.org/docs/user_guide/boolean.html)
* [datetime64[ns, tz] (numpy)](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.datetime64)
* [CategoricalDtype (pandas)](https://pandas.pydata.org/docs/reference/api/pandas.CategoricalDtype.html#pandas.CategoricalDtype)

## Data Extraction
To see the logic to extract the variables from the OMOP CDM see [h&n query](../v6-sessions/v6-sessions/sql/head_and_neck_features.sql).

### Basic variables
The variables available in both [Head and Neck](#head-and-neck) and [Sarcoma](#sarcoma).

Variable|Type|Status|Notes
--|--|--|--
`sex`|`CategoricalDtype`|✅|`concept_name` of `person.gender_concept_id`
`year_of_birth`|`Int64`|✅|`person.year_of_birth`
`diagnosis_date`|`datetime64[ns, tz]`|✅| `episode.episode_start_date` where `episode.episode_concept_id = 32533`
`age_at_diagnosis`|`Int64`|🐧|Year of `episode.episode_start_date` where `episode.episode_concept_id = 32533` - `person.year_of_birth`
`diagnosisCode`|`CategoricalDtype`|🐧|`concept_code` of `episode.episode_object_concept_id` where `episode.episode_concept_id = 32533`
`morphology`|`CategoricalDtype`|✅|First part of `concept_code` of `episode.episode_object_concept_id` where `episode.episode_concept_id = 32533`
`topography`|`CategoricalDtype`|✅|Second part of `concept_code` of `episode.episode_object_concept_id` where `episode.episode_concept_id = 32533`
`life_status`|`CategoricalDtype`|✅|`concept_name` of latest observation where `observation.observation_concept_id` is one of `2000100071`, `4230556`, `2000100072`,`2000100073`, `2000100074`, `2000100075`, `4163894`.
`life_status_date`|`datetime64[ns, tz]`|✅|`observation.observation_date` of the latest observation where `observation.observation_concept_id` is one of `2000100071`, `4230556`, `2000100072`,`2000100073`, `2000100074`, `2000100075`, `4163894`.
`surgery_intent`|`CategoricalDtype`|🐸|`concept_name` of `measurement.measurement_concept_id` where `measurement_concept_id` is one of `4179711`, `4162591` and `measurement_event_id` is equal to `procedure_occurrence.procedure_occurrence_id` which is equal to `episode_event.event_id` and `episode_event.episode_id` is equal to `episode.episode_id` where `episode.episode_concept_id = 32939`
`margins_after_surgery`|`CategoricalDtype`|🐸|`concept_name` of `measurement.measurement_concept_id` where `measurement_concept_id` is one of `1634643`, `1633801`, `1634484` and `measurement_event_id` is equal to `procedure_occurrence.procedure_occurrence_id` which is equal to `episode_event.event_id` and `episode_event.episode_id` is equal to `episode.episode_id` where `episode.episode_concept_id = 32939`
`date_of_surgery`|`datetime64[ns, tz]`|🐸||
`radio_therapy_intent`|`CategoricalDtype`|*|`concept_name` of `measurement.measurement_concept_id` where `measurement_concept_id` is one of `4179711`, `4162591`.
`radiotherapy_hospital`|???|*|
`radio_start_date`|`datetime64[ns, tz]`|*||
`radio_end_date`|`datetime64[ns, tz]`|*||
`radio_setting`|`CategoricalDtype`(?)|*|
`radio_total_dose_gy`|`Float64`|*|
`radio_number_of_fractions`|`Float64`|*|
`radio_treatment_completed_as_planned`|`CategoricalDtype`|*|
`systemic_type_of_systemic_treatment`|`CategoricalDtype`|🆕|`concept_name` of `procedure_occurrence.procedure_concept_id` where `procedure_occurrence_id` is equal to `episode_event.event_id` and `episode_event.episode_id` is equal to `episode.episode_id` where `episode.episode_concept_id = 32531`
`systemic_start_date_systemic_treatment`|`datetime64[ns, tz]`|🆕|`episode_start_date` of `episode.episode_concept_id = 32531`
`systemic_end_date_systemic_treatment`|`datetime64[ns, tz]`|🆕|`episode_end_date` of `episode.episode_concept_id = 32531`
`systemic_regimen`|`CategoricalDtype`|🆕|`episode_object_concept_id` of `episode.episode_concept_id = 32531`
`systemic_reason_for_end_of_treatment`|`CategoricalDtype`|🆕|`concept_name` of `observation.observation_concept_id` where `observation_concept_id` is one of `44788181`, `4162594`, `2000100030`, `4240582`, `37017062`, `4306655` and `observation_event_id` is equal to `procedure_occurrence.procedure_occurrence_id` which is equal to `episode_event.event_id` and `episode_event.episode_id` is equal to `episode.episode_id` where `episode.episode_concept_id = 32531`
`drugs_for_treatments`|???|🆕|`concept_name` of `drug_exposure.drug_concept_id` where `drug_exposure_start_date` is equal to `systemic_start_date_systemic_treatment` and `drug_exposure_end_date` is equal to `systemic_end_date_systemic_treatment` ⚠️ There can be multiple drugs associated with each systemic treatment, we need to define a fixed number 
`overall_treatment_response_response`|`CategoricalDtype`|*|
`overall_treatment_response_defined_done`|???||
Overall treatment response date|???| Still needs to be defined by Unai|
`clinical_is_transit_metastasis_with_clinical_confirmation`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 36769249` where `measurement.measurement_date` is equal to `diagnosis_date`
`clinical_is_multifocal_tumor`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 36769933` where `measurement.measurement_date` is equal to `diagnosis_date`
`clinical_regional_nodal_metastases`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 36769269` where `measurement.measurement_date` is equal to `diagnosis_date`
`clinical_soft_tissue`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 35225724` where `measurement.measurement_date` is equal to `diagnosis_date`
`clinical_distant_lymph_node`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 36769243` where `measurement.measurement_date` is equal to `diagnosis_date`
`clinical_lung`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 36770283` where `measurement.measurement_date` is equal to `diagnosis_date`
`clinical_metastasisatbone`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 36769301` where `measurement.measurement_date` is equal to `diagnosis_date`
`clinical_liver`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 36770544` where `measurement.measurement_date` is equal to `diagnosis_date`
`clinical_pleura`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 35226258` where `measurement.measurement_date` is equal to `diagnosis_date`
`clinical_peritoneum`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 35226253` where `measurement.measurement_date` is equal to `diagnosis_date`
`clinical_brain`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 36768862` where `measurement.measurement_date` is equal to `diagnosis_date`
`clinical_other_viscera`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 36769180` where `measurement.measurement_date` is equal to `diagnosis_date`
`clinical_unknown`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 4129922` where `measurement.measurement_date` is equal to `diagnosis_date`
`pathological_regional_nodal_metastases`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 36769269` where `measurement.measurement_date` is equal to `date_of_surgery`**
`pathological_soft_tissue`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 35225724` where `measurement.measurement_date` is equal to `date_of_surgery`**
`pathological_distant_lymph_node`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 36769243` where `measurement.measurement_date` is equal to `date_of_surgery`**
`pathological_lung`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 36770283` where `measurement.measurement_date` is equal to `date_of_surgery`**
`pathological_metastasisatbone`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 36769301` where `measurement.measurement_date` is equal to `date_of_surgery`**
`pathological_liver`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 36770544` where `measurement.measurement_date` is equal to `date_of_surgery`**
`pathological_pleura`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 35226258` where `measurement.measurement_date` is equal to `date_of_surgery`**
`pathological_peritoneum`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 35226253` where `measurement.measurement_date` is equal to `date_of_surgery`**
`pathological_brain`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 36768862` where `measurement.measurement_date` is equal to `date_of_surgery`**
`pathological_other_viscera`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 36769180` where `measurement.measurement_date` is equal to `date_of_surgery`**
`pathological_unknown`|`boolean`|🐧|`TRUE` if is present `measurement.measurement_concept_id = 4129922` where `measurement.measurement_date` is equal to `date_of_surgery`**

** Assuming that `date_of_surgery` refers to the date of the surgery performed at the primary tumor stage (first surgery), excluding any procedures related to recurrences.

### Treatment settings 🆕
Treatment|Description|Notes
--|--|--
Primary surgery|Surgery performed at the diagnosis|First episode after diagnosis in the EPISODE table where `episode_concept_id = 32939`
Neo-adjuvant/pre-operative|After the diagnosis but before primary surgery|`episode_start_date` where `episode.episode_concept_id = 32533` (diagnosis date) < treatment date < first `episode_start_date` where `episode.episode_concept_id = 32939` (primary surgery)
Adjuvant/post-operative|After surgery before recurrence|first `episode_start_date` where `episode.episode_concept_id = 32939` (primary surgery) < treatment date < first `episode_start_date` where `episode.episode_concept_id` is one of `32948`, `2000100002` (first progression or recurrence)
Recurrence|After first recurrence| treatment date > first `episode_start_date` where `episode.episode_concept_id` is one of `32948`, `2000100002` (first progression or recurrence). If we need to extract treatments for a specific recurrence, we should check that they occur between two episodes where `episode.episode_concept_id` is one of `32948`, `2000100002`

## Head and Neck variables
Variable|Type|Status|Notes
--|--|--|--
`surgery_hospital`|`CategoricalDtype`||`care_site.care_site_name`
`pathological_stage`|`CategoricalDtype`|✅|`concept_name` of `measurement.measurement_concept_id` where `measurement_concept_id` is one of  `1634741`, `1635511`, `1634787`, `1635797`, `1635800`, `1634799`, `1633751`, `1634619`, `1635386`, `1633499`, `1634947`, `1634705`, `1634208`, `1635230`, `1633697`, `1634731`, `1635745`, `1634005`, `1635370`, `1634472`, `1634487`, `1635893`, `1634492`, `1634551`.
`pathological_stage_pt`|`CategoricalDtype`|✅|`concept_name` of `measurement.measurement_concept_id` where `measurement_concept_id` is one of `1634270`, `1635402`, `1634986`, `1635660`, `1633798`, `1634720`, `1633279`, `1634675`, `1634635`, `1633445`, `1635422`, `1635070`, `1634792`, `1634491`, `1633307`, `1635670`, `1634658`, `1634386`, `1635311`, `1635341`, `1635396`, `1634101`, `1633723`, `1634894`, `1633900`, `1633699`, `1633658`.
`pathological_stage_pn`|`CategoricalDtype`|✅|`concept_name` of `measurement.measurement_concept_id` where `measurement_concept_id` is one of `1635823`, `1634505`, `1634117`, `1634212`, `1633726`, `1635560`, `1634562`, `1634245`, `1633659`, `1634541`, `1633569`, `1633336`, `1633273`, `1635717`, `1635871`, `1634773`, `1633607`, `1634645`, `1635113`, `1634601`, `1634383`, `1634504`, `1633668`, `1635307`, `1634271`, `1634397`, `1635545`, `1633500`, `1634847`, `1634770`.
`pathological_stage_pm`|`CategoricalDtype`|✅|`concept_name` of `measurement.measurement_concept_id` where `measurement_concept_id` is one of `1635345`, `1635536`, `1634606`, `1633469`, `1635336`, `1634891`.
`pathological_stage_extra_nodal_extension`|`CategoricalDtype`||`concept_name` of `measurement.measurement_concept_id` where `measurement_concept_id` is one of `36769946`,`36770618`.
`clinical_stage`|`CategoricalDtype`|✅|`concept_name` of `measurement.measurement_concept_id` where `measurement_concept_id` is one of `1635842`,  `1633828`, `1635824`, `1633905`, `1634457`, `1635758`, `1634718`, `1635182`, `1635217`, `1635848`, `1635125`, `1634596`, `1634307`, `1634766`, `1635029`, `1635535`, `1634451`, `1634810`, `1633922`, `1635757`, `1635708`, `1633270`, `1634614`, `1635006`.
`clinical_stage_ct`|`CategoricalDtype`|✅|`concept_name` of `measurement.measurement_concept_id` where `measurement_concept_id` is one of `1635299`, `1634269`, `1633589`, `1635857`, `1635227`, `1633737`, `1635656`, `1633815`, `1635794`, `1634381`, `1635664`, `1633883`, `1633747`, `1634029`, `1634651`, `1633877`, `1633324`, `1635556`, `1635522`, `1635530`, `1634973`, `1634247`, `1634522`, `1634963`, `1634624`, `1634120`, `1634854`.
`clinical_stage_cn`|`CategoricalDtype`|✅|`concept_name` of `measurement.measurement_concept_id` where `measurement_concept_id` is one of `1635104`, `1633679`, `1634797`, `1633315`, `1633942`, `1634070`, `1635697`, `1634139`, `1633651`, `1635470`, `1635634`, `1633763`, `1634143`, `1635739`, `1633788`, `1633433`, `1635677`, `1633323`, `1634678`, `1634727`, `1633271`, `1635605`, `1634037`, `1633854`, `1633434`, `1635004`, `1635496`, `1635283`, `1635084`, `1635828`.
`clinical_stage_cm`|`CategoricalDtype`|✅|`concept_name` of `measurement.measurement_concept_id` where `measurement_concept_id` is one of `1634194`, `1633468`, `1634757`, `1634829`, `1633276`, `1633974`.
`clinical_stage_extra_nodal_extension`|`CategoricalDtype`|3|`concept_name` of `measurement.measurement_concept_id` where `measurement_concept_id` is one of `36769946`, `36770618`.
`surgery_extra_nodal_extension`|`CategoricalDtype`|🆕|`concept_name` of `measurement.measurement_concept_id` where `measurement_concept_id` is one of `36769946`, `36770618` and `measurement_event_id` is equal to `procedure_occurrence.procedure_occurrence_id` which is equal to `episode_event.event_id` and `episode_event.episode_id` is equal to `episode.episode_id` where `episode.episode_concept_id = 32939`
`neck_surgery`|`CategoricalDtype`|🆕|`concept_name` of `procedure_occurrence.procedure_concept_id` where `procedure_concept_id` is equal to `4291481` and `procedure_occurrence_id` is equal to `episode_event.event_id` and `episode_event.episode_id` is equal to `episode.episode_id` where `episode.episode_concept_id = 32939` ⚠️ Maybe this can be represented as a boolean, since only one concept_id is used: `TRUE` if present and `FALSE` if not.
`laterality_of_the_dissection`|`CategoricalDtype`|🆕|`concept_name` of `measurement.measurement_concept_id` where `measurement_concept_id` is one of `4112106`, `4117496`, `4112107` and `measurement_event_id` is equal to `procedure_occurrence.procedure_occurrence_id` which is equal to `episode_event.event_id` and `procedure_occurrence.procedure_concept_id = 4291481` and `episode_event.episode_id` is equal to `episode.episode_id` where `episode.episode_concept_id = 32939`
`date_of_neck_surgery`|`datetime64[ns,tz]`|🆕|`procedure_date` of `procedure_occurrence.procedure_concept_id = 4291481` where `procedure_occurrence_id` is equal to `episode_event.event_id` and `episode_event.episode_id` is equal to `episode.episode_id` where `episode.episode_concept_id = 32939`
`radio_beam_quality`|`CategoricalDtype`|*|
`radio_total_high_dose`|`Float64`|*|
`radio_treatment_site_primary_only`|`boolean`|*|
`radio_treatment_site_neck_only`| `boolean`|*|
`radio_treatment_site_primary_and_ipsilateral_neck`|`boolean`|*|
`radio_treatment_site_primary_and_bilateral_neck`|`boolean`|*|
`radio_treatment_site_distant_metastasis`|`boolean`|*|
`systemic_intent`|`CategoricalDtype`|🆕|`concept_name` of `measurement.measurement_concept_id` where `measurement_concept_id` is one of `4179711`, `4162591` and `measurement_event_id` is equal to `procedure_occurrence.procedure_occurrence_id` which is equal to `episode_event.event_id` and `episode_event.episode_id` is equal to `episode.episode_id` where `episode.episode_concept_id = 32531`
`systemic_setting`|`CategoricalDtype`|🆕|This will be derived using dates


## Sarcoma variables
Variable|Type|Status|Notes
--|--|--|--
`type_of_biopsy`|`CategoricalDtype`|🐧|`concept_name` of <b>FIRST</b> `procedure_occurrence.procedure_concept_id` where `procedure_concept_id` is one of `4171863`, `4321878`,`4321986`, `4228202`, `4279903`, `4311405`.
`date_of_biopsy`|`datetime64[ns,tz]`|🐧|`procedure_date` of <b>FIRST</b> `procedure_occurrence.procedure_concept_id` where `procedure_concept_id` is one of `4171863`, `4321878`,`4321986`, `4228202`, `4279903`, `4311405`.
`clinical_localised`|`boolean`|🐧|`TRUE` if is present `episode.episode_concept_id = 32942` where `episode.episode_start_date` is equal to `diagnosis_date`
`clinical_number_of_tumor_nodules`|`Int64`|🐧|`value_as_number` of `measurement.measurement_concept_id = 4228659` where `measurement.measurement_date` is equal to `diagnosis_date`
`clinical_loco_regional`|`boolean`|🐧|`TRUE` if is present `episode.episode_concept_id = 32943` where `episode.episode_start_date` is equal to `diagnosis_date`
`pathelogical_localised`|`boolean`|02/04|`TRUE` if is present `episode.episode_concept_id = 32942` where `episode.episode_start_date` is equal to `date_of_surgery`**
`pathelogical_number_of_tumor_nodules`|`Int64`|02/04|`value_as_number` of `measurement.measurement_concept_id = 4228659` where `measurement.measurement_date` is equal to `date_of_surgery`**
`pathelogical_loco-regional`|`boolean`|02/04|`TRUE` if is present `episode.episode_concept_id = 32943` where `episode.episode_start_date` is equal to `date_of_surgery`**
`pathelogical_is_transit_metastasis_with_clinical_confirmation`|`boolean`|02/04|`TRUE` if is present `measurement.measurement_concept_id = 36769249` where `measurement.measurement_date` is equal to `date_of_surgery`**
`pathelogical_is_multifocal_tumor`|`boolean`|02/04|`TRUE` if is present `measurement.measurement_concept_id = 36769933` where `measurement.measurement_date` is equal to `date_of_surgery`**
`last_contact`|`datetime64[ns,tz]`|🐧|`episode.episode_end_date` where `episode.episode_concept_id = 32533`

** Assuming that `date_of_surgery` refers to the date of the surgery performed at the primary tumor stage (first surgery), excluding any procedures related to recurrences.

## Authentication
Vantage6 uses its own Keycloak instance which is linked to the CERTH keycloak instance. Authentication process will be as follows (handled by RAVEN and the keycloak instances):

```mermaid

flowchart LR

new_user@{shape: circle, label: "New\nuser"}
register@{shape: rounded, label: "User is\nregistered\nin CERTH\nkeycloak"}
stop@{ shape: framed-circle, label: "Stop" }

new_user --> register
register --> stop

login@{shape: circle, label: "RAVEN\nlogin"}
exists_raven_q@{shape: diam, label: "Username\nexists in\nRAVEN"}
raven_db@{shape: cyl, label: "RAVEN\nDB"}
exists_v6_q@{shape: diam, label: "Username\nexists in\nRAVEN"}
register_v6@{label: "Register user\nin vantage6"}
v6_db@{shape: cyl, label: "vantage6\ndatabase"}
v6_kc_registration@{label: "vantage6\nKeycloak\nregistration"}
v6_kc_db@{shape: cyl, label: "vantage6\nkeycloak\nDB"}
is_linked@{shape: diam, label: "Is vantage6\nKeycloak user\nlinked to\nCERTH user"}
kc_linking@{label: "Keycloak\nlinking process\nbased on email"}
logged_in@{shape: rounded, label: "Logged in"}

subgraph RAVEN
    login
    exists_raven_q
    raven_db
    register_v6
    exists_v6_q
end

login --> exists_raven_q
exists_raven_q -- "No" --> raven_db
raven_db --> exists_v6_q
exists_raven_q -- "✅" --> exists_v6_q
exists_v6_q -- "No" --> register_v6
register_v6 -.-> v6_db

register_v6 --> v6_kc_registration
v6_kc_registration -.-> v6_kc_db
v6_kc_registration --> is_linked

exists_v6_q -- "✅" --> is_linked
is_linked -- "No" --> kc_linking
is_linked -- "✅" --> logged_in
kc_linking --> logged_in
```

## Preprocessing Algorithms

> [!IMPORTANT]
> It is important that every preprocessing algorithm is applied to all dataframes in the same session.

### Timedelta
Compute the number it days between a `datetime` column and a reference date. The output of the function is an Int.

The input parameters are as follows:

|Argument|Required|Display as argument|Type|Description|
|---|---|---|---|---|
|`column`|✅|✅|`str`|The column name that contains the start date. This should be of type `datetime64`.
|`output_column`|✅|✅|`str`|The new variable column name.
|`to_date_column`|No|✅|`str`|The column containin the end date. If not provided the `to_date` argument will be used. If provided it the column should be of type `datetime64`.
|`to_date`|No|✅|`str`|The reference date to use `yyyy-mm-dd`. If not supplied, and the `to_date_column` is also not supplied; today will be used.

### Merge categories
Merging/grouping categories. E.g. a column has levels (=categories) `A`, `B` and `C`, but I want to group `B` and `C` into a new category `D` so that I get a new column with only categories `A` and `D`.

The input parameters are as follows:

|Argument|Required|Display as argument|Type|Description|
|---|---|---|---|---|
|`column`|✅|✅|`str`|The column name to merge categories from. The referenced column should be of type `category`.|
|`output_column`|✅|✅|`str`|The new variable column name which will contain the merged categories.|
|`mapping`|✅|✅|`dict[str, list[str]]`|A dictionary specifying how to group the levels. **Keys** are the new merged category names; **values** are lists of the original category names to merge into each new category.|

> **Example:**  
> Suppose you have a column `color` with possible values: `red`, `blue`, `green`, `yellow`. You want to group `blue` and `green` as `cool`, and keep `red` and `yellow` as their own categories.  
> Use:  
> `mapping = {"cool": ["blue", "green"]}`  
> This will create a new column where `blue` and `green` are replaced by `cool`, and `red` and `yellow` remain unchanged.

### One hot encoding
Expand a single categorical column into several columns containing the category name and a boolean value. For example if we have

|var|
|--|
|A|
|B|
|C|
|B|

After one hot encoding you will get:

|var|A|B|C|
|--|--|--|--|
|A|`True`|`False`|`False`|
|B|`False`|`True`|`False`|
|C|`False`|`False`|`True`|
|B|`False`|`True`|`False`|


The input parameters are as follows:

|Argument|Required|Display as argument|Type|Description|
|---|---|---|---|---|
|column|✅|✅|`str`|The column name to one-hot encode, the column should be of type `category`.|
|categories|✅|✅|`list[str]`|List of categories. In case there are categories present in the local data that are not in this list they will be added to the `unknown_category` group.
|unknown_category|❌|✅|`str`|The column/variable name for categories which are present in the date but not in argument list of `categories`|
|prefix|❌|✅|`str`|Prefix for the new one-hot encoded columns|

### Merge variables
Combine two categorical variables into one. For example combine columns `A` and `B` into a new column `C`:

A|B
--|--
Q1|G2
Q3|G4

becomes:

|A|B|C|
--|--|--
Q1|G2|Q1_G2
Q3|G4|Q3_G4

The input parameters are as follows:

|Argument|Required|Display as argument|Type|Description|
|---|---|---|---|---|
|`column1`|✅|✅|`str`|The first column name to use in the merge, this column needs to be of `category` type.
|`column2`|✅|✅|`str`|The second column to use in the merge, this column needs to be of the `category` type.
|`output_column`|✅|✅|`str`|The name of the new column.

### Basic Arithmetic
Basic operations: `add`, `subtract`, `multiply` and `divide`. All these operations can work on pandas series and on numeric values. 

|Argument|Required|Display as argument|Type|Description|
|---|---|---|---|---|
|`column1`|✅|✅|`str` or `int` or `float`|The column name or value of the left hand side of the operation. In the case of a column name, the type of the column should either be `Int64` or `Float64`.
|`column2`|✅|✅|`str` or `int` or `float`|The column name or value of the right hand side of the operation. In the case of a column name, the type of the column should either be `Int64` or `Float64`.
|`operation`|✅|✅|`str`| The operation to perform, should be one of `add`, `subtract`, `multiply`, `divide`.
|`output_column`|✅|✅|`str`|The name of the new column.


### Drop column
You might want to drop a column in case you created a column that was not correct. This allows you to delete one.

The input parameters are as follows:

|Argument|Required|Display as argument|Type|Description|
|---|---|---|---|---|
|`column`|✅|✅|`str`|The column name to drop.


## Analytics Algorithms 

### Summary and Table1
The summary algorithm computes a lot of descriptive statistics. I suggest to have a
brief look at the swimlane diagram in the [Security and Privacy documentation](./security-and-privacy/summary.pdf).

>[!NOTE]
> This algorithm is both used to compute the statistics for the data preparation step and to compute the table1 output.

The input parameters are as follows:

|Argument|Required|Display as argument|Type|Description|
|---|---|---|---|---|
|`columns`|No|No|`list[str]`| List of variable names to compute the summary for. If omnitted, what should be the case in IDEA4RC, all variables available in the dataframe are analysed.
|`numeric_colums`|No|No|`list[str]`| List of variables that are numerical, should be a subset of the `columns` and the column type should be numerical. If omnitted, what should be the case in IDEA4RC, types are inferred.|
|`organizations_to_include` |No|No| `list[int]` | List of vantage6 organization IDs that need to be included in the analysis|
|`stratification_column`|No|✅|`str`| Name of the variable that the results should be stratified to. In case of the data preperation this should be omnitted. In case of the table1 analysis this should be an optional parameter that the user can specify. **The stratification variable should be of type `categorical`**.|

### Crosstabs & Chi-Square (Contingency table)
The crosstabs algorithm computes the contingency table of two or more categorical
variables. I suggest to have a brief look at the swimlane diagram in the [Security and Privacy documentation](./security-and-privacy/Security%20&%20Privacy%20Crosstab.pdf).

The input parameters are as follows:

|Argument|Required|Display as argument|Type|Description|
|---|---|---|---|---|
|`results_col`| ✅ |✅| `str` | The variable for which counts are calculated. **The variable should be of type `categorical`**.
| `groups_col` | ✅ |✅| `list[str]` | List of variables to group the data by. **Each of the variables in the list should be of type `categorical`**
| `organizations_to_include` |No|No| `list[int]` | List of vantage6 organization IDs that need to be included in the analysis|
| `include_chi2` | No |No| `bool` | Do not supply as this is by default `True` | 
| `include_totals` | No |No| `bool` | Do no supply as this is by default `True` | 

### T-test
The T test algorithm computes a t-test for two (or more) samples. I suggest to have a brief look at the swimlane diagram in the [Security and Privacy documentation](./security-and-privacy/Security%20&%20Privacy%20t-test.pdf) to have a good overview of the different steps in the algorithm.

The t-test will automatically use all available variables of the type *numeric*, therefore there is no option to specify which variables to compute it for. This implementation computes the t-test of all numeric variables between **2 organizations*. If needed we can extend this to the case that the user selects a single reference organization to which we compute all t-test compared to that single organization.

The input parameters are as follows:

|Argument|Required|Display as argument|Type|Description|
|---|---|---|---|---|
| `organizations_to_include` |No|✅(!)| `list[int]` | List of vantage6 organization IDs that need to be included in the analysis. **In the case of the t-test we need this to be exactly two organizations!** |

### Kaplan Meier and Log Rank
The Kaplan Meier algorithm computes a Kaplan Meier estimate for the survival function. I suggest to have a brief look at the swimlane diagram in the [Security and Privacy documentation](./security-and-privacy/Security%20&%20Privacy%20Kaplan-Meier.pdf) to have a good overview of the different steps in the algorithm.

|Argument|Required|Display as argument|Type|Description|
|---|---|---|---|---|
|`time_column_name`|✅|✅|`str`| The variable name which contains the (survival) time. **This variable needs to be of type `numeric`**.
|`censor_column_name`|✅|✅|`str`|The variable name which contains the (survival) time. **This variable needs to be of type `bool`**.
|`organizations_to_include`|No|No|`list[int]`|List of vantage6 organization IDs that need to be included in the analysis.|
|`strata_column_name`|No|✅|`str`| The variable name to which you want to stratify. **This variable needs to be of the type `categorical`**|

### GLM 
A GLM (Generalized Linear Model) lets you model relationships between variables using linear predictors and flexible distributions, enabling regression and classification tasks. I sugest to have a brief look at the swimlane diagram in the [Security and Privacy documentation](./security-and-privacy/Security%20&%20Privacy%20GLM.pdf) to have a goof overview of the different steps in the algorithm.

|Argument|Required|Display as argument|Type|Description|
|---|---|---|---|---|
|`family`|✅|✅|`str`|The exponential family to use for computing the GLM. The available families are `Gaussian`, `Poisson`, `Binomial`, and `Survival`. Depening on which family you select the type of the `outcome_variable` should be different, see the table bellow|
|`outcome_variable`|✅|✅|`str`| The variable name of the outcome variable. Not that the type of this variable is dependent on the `family` argument. To see this relationship see the table bellow.|
|`predictor_variables`|✅|✅|`list[str]`| The variable names of the predictor variables. The only type that is not allowed here is `datetime[ns, tz]`. `numeric`, `bool` and `category` are accepted.|
|`formula`|No|No|`str`|A text based formula for extra flexibility, I recommend hiding this at least for now|
|`categorical_predictors`|No|No|`list[str]`|The column names of the predictor variables that are categorical. In IDEA4RC we do not supply this as the types are clearly defined in the local datasets, so the algorithm can infer them.
|`category_reference_values`|No|✅|`dict[str, str]`|A dictonairy that contains variable names as keys and the reference value as the value. *For now we do not know which levels each category has, thus the values need to be able to specify as a free text field*. The variable names that are supplied as keys, need to be of type `category`.|
|`survival_sensor_column`|No*|No*|`str`|The variable name of the survival censor. *Required if the `family` is set to `Survival`. The type of the variable should be `bool`.
|`tolerance_level`|No|No|`numeric`|Do not supply as the default value should be used for now|
|`max_iterations`|No|No|`numeric`|Do not supply as the default value should be used for now|
| `organizations_to_include` |No|No| `list[int]` | List of vantage6 organization IDs that need to be included in the analysis|

### CoxPH
The Cox algorithm looks at how different factors (like age or treatment) change a person’s chance of experiencing something over time, for example dying or having a complication. It tells you whether a factor seems to increase, decrease, or not really affect that chance, while using all the follow-up time information instead of just "event yes/no".

TODO: The security an privacy document

|Argument|Required|Display as argument|Type|Description|
|---|---|---|---|---|
|`time_col`|✅|✅|`str`|The variable name of the time column. This should be a numerical, either `Float64` or `Int64`, column that represents the survival time.
|`outcome_col`|✅|✅|`str`|The variable name of the outcome column. This should be a `boolean` variable.
|`expl_vars`|✅|✅|`list[str]`|The variable names that represent the covariates. They should be numerical, either `Float64` or `Int64`, columns.
|`organizations_to_include`|No|No|`list[int]`|List of vantage6 organization IDs that need to be included in the analysis|
