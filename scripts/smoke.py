#!/usr/bin/env python3
# scripts/smoke.py
"""
Ultra-safe pre-deploy smoke test for Providers apps.
- Confirms SQL connectivity (Turso if secrets present; else local SQLite).
- Verifies vendors/categories/services tables exist.
- Prints row counts; exits non-zero on failure.
No code changes to apps are required.
"""
from __future__ import annotations
import os, sys
from typing import Dict, Tuple
import pandas as pd
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

def _get_secret(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)

def _make_engine() -> Tuple[Engine, Dict[str, str]]:
    status: Dict[str, str] = {}
    url = _get_secret("TURSO_DATABASE_URL")
    token = _get_secret("TURSO_AUTH_TOKEN")
    if url and token:
        status["backend"] = "libsql"
        status["dsn"] = "sqlite+libsql://<redacted-host>"
        eng = create_engine(url, connect_args={"auth_token": token})
    else:
        status["backend"] = "sqlite"
        status["dsn"] = "sqlite:///vendors.db"
        eng = create_engine("sqlite:///vendors.db")
    return eng, status

def main() -> int:
    eng, status = _make_engine()
    print({"db": status})
    try:
        with eng.connect() as conn:
            conn.execute(sql_text("SELECT 1"))
    except Exception as e:
        print({"error": f"connect failed: {type(e).__name__}: {e}"})
        return 2

    required = ("vendors", "categories", "services")
    missing, counts = [], {}
    try:
        with eng.connect() as conn:
            for tbl in required:
                try:
                    cols = pd.read_sql(sql_text(f"PRAGMA table_info({tbl})"), conn)["name"].tolist()
                    if not cols:
                        missing.append(tbl); continue
                    c = pd.read_sql(sql_text(f"SELECT COUNT(*) AS c FROM {tbl}"), conn)["c"].iat[0]
                    counts[tbl] = int(c)
                except Exception:
                    missing.append(tbl)
    except Exception as e:
        print({"error": f"probe failed: {type(e).__name__}: {e}"})
        return 3

    print({"tables": {"missing": missing, "counts": counts}})
    if missing:
        print("SMOKE: FAIL — missing tables:", ", ".join(missing)); return 4
    print("SMOKE: OK"); return 0

if __name__ == "__main__":
    sys.exit(main())
