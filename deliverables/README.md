# IDEA4RC Deliverables

This folder contains the deliverables required for the integration with the RAVEN UI.
Each algorithm has its own folder, containing the following files:

- Security an Privacy document
- API documentation for the lifecycle of the algorithm
- Visualization documentation for the analytics


## Workflow
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

* [numeric (numpy)](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.number)
* [bool (numpy)](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool_)
* cate
* [datetime64[ns, tz] (numpy)](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.datetime64)
* [CategoricalDtype (pandas)](https://pandas.pydata.org/docs/reference/api/pandas.CategoricalDtype.html#pandas.CategoricalDtype)


