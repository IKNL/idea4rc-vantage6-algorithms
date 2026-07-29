#!/usr/bin/env python3
"""Render features.sql.j2 for both variants and check their SQL syntax.

Renders the Jinja2 template at ``v6-sessions/v6-sessions/sql/features.sql.j2``
for the ``head_and_neck`` and ``sarcoma`` variants (exactly the way
``cohort.py`` does), writes each rendered statement next to this script, and
parses it with sqlglot in the ``postgres`` dialect to catch syntax errors such
as missing commas between SELECT items.

Run it with uv (no need to add the tooling deps to the package):

    uv run --with sqlglot --with jinja2 python check_features_sql.py

Exit code is 0 when every variant parses, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import sqlglot
from jinja2 import Template

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "v6-sessions" / "sql" / "features.sql.j2"
OUTPUT_DIR = HERE / "rendered"

# The variants map onto the boolean flags cohort.py passes to render().
VARIANTS = ("head_and_neck", "sarcoma")

# A small, representative patient list — the id list does not affect syntax.
PATIENT_IDS = ", ".join(f"({i})" for i in range(1, 6))
CDM_SCHEMA = "cdm_idea"


def render(template: Template, variant: str) -> str:
    """Render the template for one variant, mirroring cohort.py's call."""
    return template.render(
        patient_ids=PATIENT_IDS,
        cdm_schema=CDM_SCHEMA,
        is_head_and_neck=(variant == "head_and_neck"),
        is_sarcoma=(variant == "sarcoma"),
    )


def check(variant: str, sql: str) -> bool:
    """Parse the rendered SQL; return True on success, print details on error."""
    try:
        sqlglot.parse_one(sql, read="postgres")
    except Exception as exc:  # sqlglot raises ParseError / TokenError subclasses
        print(f"[FAIL] {variant}: syntax error")
        for line in str(exc).splitlines():
            print(f"       {line}")
        return False
    print(f"[ OK ] {variant}: parsed cleanly")
    return True


def main() -> int:
    if not TEMPLATE.exists():
        print(f"Template not found: {TEMPLATE}", file=sys.stderr)
        return 2

    template = Template(TEMPLATE.read_text())
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_ok = True
    for variant in VARIANTS:
        sql = render(template, variant)
        out_path = OUTPUT_DIR / f"features_{variant}.sql"
        out_path.write_text(sql)
        print(f"Rendered {variant} -> {out_path.relative_to(HERE)}")
        all_ok &= check(variant, sql)

    print("-" * 60)
    print("All variants parsed." if all_ok else "One or more variants failed.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
