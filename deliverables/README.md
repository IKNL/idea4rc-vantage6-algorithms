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
* [datetime64[ns, tz] (numpy)](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.datetime64)
* [CategoricalDtype (pandas)](https://pandas.pydata.org/docs/reference/api/pandas.CategoricalDtype.html#pandas.CategoricalDtype)


## Authentication
Vantage6 uses its own Keycloak instance which is linked to the CERTH keycloak instance. Authentication process will be as follows (handled by RAVEN and the keycloak instances):

```mermaid

flowchart TB

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
exists_raven_q -- "Yes" --> exists_v6_q
exists_v6_q -- "No" --> register_v6
register_v6 -.-> v6_db

register_v6 --> v6_kc_registration
v6_kc_registration -.-> v6_kc_db
v6_kc_registration --> is_linked

exists_v6_q -- "Yes" --> is_linked
is_linked -- "No" --> kc_linking
is_linked -- "Yes" --> logged_in
kc_linking --> logged_in
```
