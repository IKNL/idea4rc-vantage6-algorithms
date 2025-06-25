# Vantage6 Analytics

These are the algorithms used to analyze the data extracted by the `v6-sessions`
algorithm.

## How to build and push the image
```bash
make image
```

In case you want to push it to the registry, you can do so with:

```bash
make image PUSH_REG=true
```

This pushes the image to the `harbor2.vantage6.ai/idea4rc/analytics` registry. It tags
this image both with `latest` and the version number in the format
`${TAG}-v6-${VANTAGE6_VERSION}`. Both the `TAG` and `VANTAGE6_VERSION` need to be set
in the `Makefile`.

```bash
make image TAG=v1.0.0 VANTAGE6_VERSION=5.0.0a19
```

(This can later be done by the CI/CD pipeline.)