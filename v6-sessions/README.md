# Vantage6 Sessions

This is a data extraction algorithm for the vantage6 platform. It connects to an OMOP
database and extracts data for a list of patient IDs. The extraction job both works
for the sarcoma and head and neck features.

In the node, the following environment variables need to be set:

- `DATABASE_URI`: The URI of the OMOP database. This should be a valid OHDSI connection
  string. For example:
  ```
  jdbc:postgresql://localhost:5432/omop
  ```
- `DATABASE_TYPE`: The type of the database. This should be OMOP, but it will not be
  used in this algorithm.
- `DATABASE_USER`: The user of the database. This user only requires read access to the
  database.
- `DATABASE_PASSWORD`: The password of the database.

## How to build and push the image
```bash
make image
```

In case you want to push it to the registry, you can do so with:

```bash
make image PUSH_REG=true
```

This pushes the image to the `harbor2.vantage6.ai/idea4rc/sessions` registry. It tags
this image both with `latest` and the version number in the format
`${TAG}-v6-${VANTAGE6_VERSION}`. Both the `TAG` and `VANTAGE6_VERSION` need to be set
in the `Makefile`.

```bash
make image TAG=v1.0.0 VANTAGE6_VERSION=5.0.0a19
```

(This can later be done by the CI/CD pipeline.)