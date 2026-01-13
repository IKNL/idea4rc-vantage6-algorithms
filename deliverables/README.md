# IDEA4RC Deliverables

This folder contains the deliverables required for the integration with the RAVEN UI.
Each algorithm has its own folder, containing the following files:

- Security an Privacy document
- API documentation for the lifecycle of the algorithm
- Visualization documentation for the analytics

## Algorithms
.. TODO basic algorithm docs ..


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
