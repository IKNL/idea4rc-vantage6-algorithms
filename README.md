# IDEA4RC Vantage6 Analytics

This repository is home to all IDEA4RC custom vantage6 code. It contains the following
important things:

- The required algorithms. It contains two algorithms:
    - `v6-analytics` (containing all our standard analytics algorithms)
    - `v6-sessions` (containing the data extraction algorithm)
- The `dev_notebooks/raven-api.ipynb` notebook contains the code that the people from
  IDEA4RC use to integrate the vantage6 API into their RAVEN UI.

## Overall Architecture

In IDEA4RC there are two main components:

- The Orchestrator (home to the vantage6 server, RAVEN UI and some other services)
- The Capsule (home to the algorithms and the data)

.. DATA (OMOP - IDEA4RC - FHIR)
.. RAVEN UI
.. VANTAGE6 UI
.. OTHER vantage6 components

### Orchestrator

The Orchestrator contains the vantage6 server, store (and later the UI). It is deployed
by `@Alejandro Alonso López`, `@Daniele Pavia` and me (`@Frank Martin`). It is setup in
a primitive way, and you can some file in the home directory of `daniele`.

In case you need to access the Orchestrator, you should contact
`@Alejandro Alonso López`.

### Capsule

The Capsule contains a lot of services we do not care about. `@Daniele Pavia` is the
one responsible for the Capsule but we need to help with the integrations of the helm
charts from vantage6.

`@Daniele Pavia` has a server with a test capsule, which we are trying to connect to the
Orchestrator.

## Project Partners
These are the most important *technical* people involved in the project, from our side:

- Architecture: `@Eugenio Gaeta` (UPM)
- RAVEN UI: `@Itziar Alonso`, `@Alejandro Alonso López` (Implementation) and
  `@Laura Lopez Perez` (Design and User stories) (UPM)
- KeyCloak: `@Athanasios Patenidis` (CERTH)
- Capsule Helm chart: `@Daniele Pavia` (UPM)

## Status of the project
The vantage6 infrastructure has all the required major components (ofc. some smaller
changes need to be done). So the deployment is now in progress. I hope I can finish
it before leaving on holiday.

### Synthetic Data

.. TODO store the synthetic data in the right place.
.. TODO how to generate the MockData
.. TODO show the IDEA4RC data model.

### Algorithms

The algorithms are all copied into this repository, I know horrible. But this way we
can keep it flexible, and hopefully we can merge it back to the official vantage6
repository. But for now, we are using it as a local repository. The status of the
algorithms:

- `v6-analytics/summary.py` is working with the mock client
- `...`

The status of the session algorithm is:
- `v6-sessions/cohort.py:create_cohort` is working with the mock client

> [!NOTE]
> In order to run the `create_cohort` function from `v6-session` locally you need
> install the OHDSI R packages: [SqlRender](https://ohdsi.github.io/SqlRender/),
> [DatabaseConnector](https://ohdsi.github.io/DatabaseConnector/) and
> [FeatureExtraction](https://ohdsi.github.io/FeatureExtraction/).


I use the `make image PUSH_REG=true` to build and push the image to the
`harbor2.vantage6.ai/idea4rc` registry. (So no build pipelines yet).








### Use Cases


