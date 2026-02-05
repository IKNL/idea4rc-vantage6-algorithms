<img src="https://github.com/IKNL/guidelines/blob/master/resources/logos/iknl_nl.png?raw=true" width=200 align="right">

# IDEA4RC custom vantage6 algorithms and documentation

This repository is home to all IDEA4RC custom vantage6 code. It contains the following
important things:

- The required algorithms. It contains two algorithms:
    - `v6-analytics` (containing all our standard analytics algorithms)
    - `v6-sessions` (containing the data extraction algorithm)
    - `v6-preprocessing` (containing the preprocessing algorithms)
- The folder [`deliverables`](/deliverables/) contains all resources for the integration with vantage6/

Each of the algorithm packages contains a `Makefile` to help with pipeline action, e.g.: releasing a new version of the algorithms.

