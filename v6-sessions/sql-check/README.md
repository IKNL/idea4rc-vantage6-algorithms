# SQL syntax check

`check_features_sql.py` renders [`../v6-sessions/sql/features.sql.j2`](../v6-sessions/sql/features.sql.j2)
for both feature variants (`head_and_neck` and `sarcoma`) the same way
[`cohort.py`](../v6-sessions/cohort.py) does, writes each rendered statement to
`rendered/features_<variant>.sql`, and parses it with
[sqlglot](https://github.com/tobymao/sqlglot) in the `postgres` dialect.

This catches pure SQL **syntax** errors — e.g. a missing comma between SELECT
items — before the query is ever sent to a node's database. It does **not**
validate semantics (column/table names, joins, CTE ordering).

## Run

```bash
uv run --with sqlglot --with jinja2 python check_features_sql.py
```

Exit code `0` means every variant parsed; `1` means at least one failed (details
printed). The `rendered/` output is regenerated on each run and is safe to
delete.
