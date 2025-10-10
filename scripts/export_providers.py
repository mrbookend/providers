#!/usr/bin/env python3
# scripts/export_providers.py
"""
Exports providers to backups/providers-YYYY-MM-DD HHMM.csv
- Uses Turso if TURSO_* env present, else local SQLite vendors.db
- Safe: read-only select, writes CSV under /backups
"""
from __future__ import annotations
import os, sys, datetime, pathlib
import pandas as pd
from sqlalchemy import create_engine, text as sql_text

def _make_engine():
    url = os.getenv("TURSO_DATABASE_URL")
    token = os.getenv("TURSO_AUTH_TOKEN")
    if url and token:
        return create_engine(url, connect_args={"auth_token": token})
    return create_engine("sqlite:///vendors.db")

def main() -> int:
    engine = _make_engine()
    q = """
        SELECT id, category, service, business_name, contact_name, phone, address, website,
               notes, keywords, created_at, updated_at, updated_by
        FROM vendors
        ORDER BY lower(trim(category)), lower(trim(service)), lower(trim(business_name))
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(sql_text(q), conn)
    except Exception as e:
        print(f"Export failed (DB): {e}"); return 2

    backups_dir = pathlib.Path("backups")
    backups_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H%M")
    fn = backups_dir / f"providers-{ts}.csv"
    try:
        df.to_csv(fn, index=False)
    except Exception as e:
        print(f"Export failed (write): {e}"); return 3

    print(f"Wrote {fn} with {len(df)} rows")
    return 0

if __name__ == "__main__":
    sys.exit(main())
