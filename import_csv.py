#!/usr/bin/env python3
# import_csv.py
# -*- coding: utf-8 -*-
"""
Import vendors CSV into Turso (libsql) using SQLAlchemy.
- Default mode: append (INSERT OR IGNORE by id)
- Optional: --mode replace (INSERT OR REPLACE by id)
- Validates headers and creates table if missing.
- Prints import count, total count, and a 5-row sample.

Env:
  * Preferred: LIBSQL_URL_FULL="libsql://<host>?authToken=<JWT>&tls=true"
  * Or: HOST="<host>" and TOKEN="<jwt>"  (tls is forced on)

Usage examples:
  python3 import_csv.py vendors_seed.csv
  python3 import_csv.py vendors_seed.csv --mode replace
"""

import os, sys, csv, re, argparse
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import create_engine, text as T

REQUIRED = [
    "id","category","service","business_name",
    "contact_name","phone","address","website",
    "notes","keywords","computed_keywords",
    "ckw_locked","ckw_version","updated_by"
]

# Minimal superset schema (types chosen to be sqlite-friendly & future-proof)
DDL = """
CREATE TABLE IF NOT EXISTS vendors (
  id                INTEGER PRIMARY KEY,
  category          TEXT NOT NULL,
  service           TEXT NOT NULL,
  business_name     TEXT NOT NULL,
  contact_name      TEXT,
  phone             TEXT,              -- store digits only; format in UI
  address           TEXT,
  website           TEXT,
  notes             TEXT,
  keywords          TEXT,
  computed_keywords TEXT,
  ckw_locked        INTEGER DEFAULT 0, -- 0/1
  ckw_version       TEXT,
  created_at        TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  updated_at        TEXT,
  updated_by        TEXT
);
"""

def _libsql_url() -> str:
    full = (os.getenv("LIBSQL_URL_FULL") or "").strip()
    if full.startswith("libsql://") and "authToken=" in full:
        return full
    host = (os.getenv("HOST") or "").strip()
    tok  = (os.getenv("TOKEN") or "").strip()
    if not host or not tok:
        print("ERROR: Provide LIBSQL_URL_FULL or HOST + TOKEN in env.", file=sys.stderr)
        sys.exit(2)
    # Force TLS
    return f"libsql://{host}?authToken={tok}&tls=true"

def _build_engine():
    url = _libsql_url()
    dsn = f"sqlite+libsql:///?url={url}"
    return create_engine(dsn, pool_pre_ping=True, pool_recycle=300)

def _digits_only(s: str) -> str:
    return re.sub(r"\D+", "", s or "")

def _sanitize_url(s: str) -> str:
    s = (s or "").strip()
    if not s: return ""
    # basic guard; leave full normalization to the apps
    if not re.match(r"^https?://", s, re.I):
        s = "https://" + s
    return s

def _ensure_table(engine):
    with engine.begin() as cx:
        cx.exec_driver_sql(DDL)

def _validate_headers(df: pd.DataFrame):
    cols = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        print(f"ERROR: CSV missing required columns: {missing}", file=sys.stderr)
        sys.exit(3)

def _normalize_row(row: dict) -> dict:
    # Normalize phone and website; leave other normalization to app
    row = dict(row)
    row["phone"] = _digits_only(row.get("phone",""))
    row["website"] = _sanitize_url(row.get("website",""))
    # updated_at default now if blank; updated_by passthrough
    if not (row.get("updated_at") or "").strip():
        row["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return row

def import_csv(engine, csv_path: str, mode: str) -> int:
    df = pd.read_csv(csv_path)
    _validate_headers(df)

    # Keep only required columns + tolerate extras by dropping them
    df = df[[c for c in df.columns if c in REQUIRED]].copy()

    rows = [ _normalize_row(r._asdict() if hasattr(r, "_asdict") else r.to_dict())
             for _, r in df.iterrows() ]

    verb = "INSERT OR IGNORE" if mode == "append" else "INSERT OR REPLACE"
    cols = REQUIRED
    placeholders = ", ".join([f":{c}" for c in cols])
    sql = f"{verb} INTO vendors ({', '.join(cols)}) VALUES ({placeholders})"

    with engine.begin() as cx:
        # batch insert
        cx.execute(T(sql), rows)

    return len(rows)

def total_count(engine) -> int:
    with engine.connect() as cx:
        return int(cx.execute(T("SELECT COUNT(*) FROM vendors")).scalar() or 0)

def sample_rows(engine, limit=5):
    with engine.connect() as cx:
        q = """
            SELECT id, category, service, business_name
              FROM vendors
             ORDER BY id
             LIMIT :n
        """
        return cx.execute(T(q), {"n": limit}).fetchall()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to CSV (e.g., vendors_seed.csv)")
    ap.add_argument("--mode", choices=["append","replace"], default="append",
                    help="append = INSERT OR IGNORE; replace = INSERT OR REPLACE (by id)")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    eng = _build_engine()
    _ensure_table(eng)
    n = import_csv(eng, args.csv, args.mode)
    tot = total_count(eng)
    samp = sample_rows(eng, 5)

    print(f"IMPORT_ROWS: {n}")
    print(f"TOTAL_NOW: {tot}")
    print(f"SAMPLE: {samp}")

if __name__ == "__main__":
    main()
