"""Schema loading helpers (shared)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from sqldecomp.model import Column, Schema, Table, schema_from_json, schema_to_json

# Oracle type of the BEAVER file -> the Postgres type that the beaver_dw load really has (checked against
# information_schema.columns). Component 8 compares these names: an equality counts only
# between columns of one comparison family, so CHAR must stay CHAR (Postgres compares char with text as bpchar, where
# 'a' = 'a ') and NUMBER is an exact numeric, not a double. The four types that occur in BEAVER are VARCHAR2, NUMBER,
# DATE and CHAR; the others are the closest Postgres type. An Oracle type that is not listed stays "unknown".
_BEAVER_TYPES = {"VARCHAR2": "TEXT", "NVARCHAR2": "TEXT", "CHAR": "CHAR", "CLOB": "TEXT",
                 "NUMBER": "NUMERIC", "FLOAT": "DOUBLE PRECISION", "INTEGER": "NUMERIC", "INT": "NUMERIC",
                 "DATE": "DATE", "TIMESTAMP": "TIMESTAMP"}


def load_schema(path: str | Path) -> Schema:
    """A schema in the contract's own JSON format (see benchmarks/toy/schema.json)."""
    return schema_from_json(json.loads(Path(path).read_text()))


def load_beaver_schema(path: str | Path) -> Schema:
    """The BEAVER file format: a list of {db, table_name, column_names, column_types}. It declares no constraints."""
    tables = []
    for t in json.loads(Path(path).read_text()):
        cols = tuple(Column(c.lower(), _BEAVER_TYPES.get(re.sub(r"\(.*\)", "", str(ty)).strip().upper(), "unknown"))
                     for c, ty in zip(t["column_names"], t["column_types"]))
        tables.append(Table(t["table_name"].lower(), cols))
    return Schema(tuple(tables))


def schema_json_of(path: str | Path) -> dict:
    """The schema JSON that the texts of component 9 and of the verifier render, and that their canonical keys and the
    run record's schema version hash: a file in the contract's format as written (its declared case, its `join_pairs`,
    every field it holds), a file in the BEAVER format through the schema model (`schema_to_json(load_beaver_schema)`).
    Every entry point that writes tasks or verifier prompts takes the schema text from it, so one schema file gives
    one text everywhere; the importer and the other components read the schema model (`load_schema`,
    `load_beaver_schema`), whose names are folded to lower case."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return schema_to_json(load_beaver_schema(path))
    if not isinstance(data, dict) or not isinstance(data.get("tables"), list):
        raise ValueError(f"{path}: not a schema (an object with `tables`, or the BEAVER list of tables)")
    return data
