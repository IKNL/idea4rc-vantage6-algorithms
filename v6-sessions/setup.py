from os import path
from codecs import open
from setuptools import setup, find_packages

# we're using a README.md, if you do not have this in your folder, simply
# replace this with a string.
here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Here you specify the meta-data of your package. The `name` argument is
# needed in some other steps.
setup(
    name="v6-sessions",
    version="1.0.0",
    description=(
        "This will extract a dataset from a set of patient IDs from the OMOP database"
        "and returns a dataframe used by vantage6 to create a dataframe."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/idea4rc/v6-sessions",
    packages=find_packages(),
    python_requires=">=3.13",
    install_requires=[
        "vantage6-algorithm-tools==5.0.0a41",
        "pandas",
        "pyarrow",
        "parquet-tools",
        "rpy2",
        "numpy==1.26.4",
        "ohdsi-common",
        "ohdsi-database-connector",
        "ohdsi-sqlrender",
        "setuptools",
    ],
    package_data={"v6-sessions": ["sql/*.sql"]},
)
