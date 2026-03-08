# SPECCHIO Access Notes

Source reviewed on 2026-03-08:

- [Python_accessing_SPECCHIO.pdf](https://github.com/SPECCHIODB/Guides/blob/master/Programming%20Course/Python_accessing_SPECCHIO.pdf)

## What the guide shows

SPECCHIO is not presented as a simple public file-download portal. The official
Python workflow uses the SPECCHIO Java client from Python through a bridge:

- `JPype` is the preferred bridge
- `Py4J` is shown as an alternative

The guide starts a JVM with `specchio-client.jar`, creates a
`SPECCHIOClientFactory`, gets server descriptors, and creates a client
connection. Data access is then query-driven:

- build a query object
- request matching spectrum ids
- create spectral spaces
- call `loadSpace(...)`
- read vectors and wavelengths from the loaded space

## Implication for this repository

`specchio_portal` should remain a portal-level manifest entry, not a direct
download source. To ingest SPECCHIO content into the spectral database, we need:

- a concrete SPECCHIO server/descriptor target
- the SPECCHIO client jar available in the build environment
- a query definition for the desired campaign or dataset
- campaign-level manifest records instead of one generic portal record

## Current adapter shape

The repository now has a `specchio_client` fetch adapter intended for
campaign-level exports. It is environment-driven so GitHub Actions or local
runs can inject the connection details at runtime:

- `SPECCHIO_CLIENT_JAR`
- `SPECCHIO_JVM_PATH` (optional)
- `SPECCHIO_JAVA_HOME` (optional)
- `SPECCHIO_SERVER_INDEX` (defaults to `0`)
- `SPECCHIO_ORDER_BY` (defaults to `Acquisition Time`)
- `SPECCHIO_QUERY_JSON` for asset exports

`SPECCHIO_QUERY_JSON` can be either:

- an inline JSON list of `{attribute, operator, value}` objects
- a path to a JSON file containing either that list or `{"conditions": [...]}`

In metadata mode the adapter writes a `specchio_runtime.json` file with the
resolved descriptor information. In assets mode it loads the matching spaces and
exports them as `space_###.csv` files.
