# app_readonly.py
# -*- coding: utf-8 -*-
from __future__ import annotations

# ---- Page config MUST be first streamlit call ----
import streamlit as st
st.set_page_config(page_title="HCR Providers — Read-Only", page_icon="📇", layout="wide")

import os, re, sys, time
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# Ensure the libsql dialect is registered (sqlite+libsql)
try:
    import sqlalchemy_libsql  # noqa: F401
except Exception:
    pass

APP_VER = "readonly-2025-10-16.1"

def _get_secret(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

def _as_bool(v, default=False) -> bool:
    if isinstance(v, bool): return v
    if v is None: return default
    return str(v).strip().lower() in ("1","true","t","yes","y","on")

SHOW_STATUS = _as_bool(_get_secret("READONLY_SHOW_STATUS", True), True)
SHOW_DIAGS  = _as_bool(_get_secret("READONLY_SHOW_DIAGS", False), False)
DB_STRATEGY = str(_get_secret("DB_STRATEGY", "embedded_replica")).strip().lower()
TURSO_URL   = _get_secret("TURSO_DATABASE_URL", "")
TURSO_TOKEN = _get_secret("TURSO_AUTH_TOKEN", "")
EMBEDDED_PATH = _get_secret("EMBEDDED_DB_PATH", "vendors-embedded.db")

def _now_utc_iso():
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

@st.cache_resource(show_spinner=False)
def build_engine() -> Engine:
    if DB_STRATEGY == "embedded_replica":
        path = EMBEDDED_PATH if os.path.isabs(EMBEDDED_PATH) else os.path.join(os.getcwd(), EMBEDDED_PATH)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        url = f"sqlite+libsql:///{path}"
    elif DB_STRATEGY == "turso_only":
        url = TURSO_URL
    else:
        url = f"sqlite+libsql:///{EMBEDDED_PATH}"
    return create_engine(url, pool_pre_ping=True, pool_recycle=300)

def _get_data_version(engine: Engine) -> str:
    try:
        with engine.connect() as cx:
            return str(cx.execute(sql_text("SELECT value FROM meta WHERE key='data_version'")).scalar() or "0")
    except Exception:
        return "0"

@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def load_df(_engine: Engine, version: str) -> pd.DataFrame:
    q = """
    SELECT id, business_name, category, service, phone, website, notes, computed_keywords
      FROM vendors
     WHERE deleted_at IS NULL
     ORDER BY business_name COLLATE NOCASE
    """
    with _engine.connect() as cx:
        df = pd.read_sql(sql_text(q), cx)
    return df


def _fmt_phone(d: str) -> str:
    d = "".join(ch for ch in (d or "") if ch.isdigit())
    if len(d) == 10:
        return f"({d[:3]}) {d[3:6]}-{d[6:]}"
    return d


def _filter(df: pd.DataFrame, q: str) -> pd.DataFrame:
    qn = (q or "").strip().lower()
    if not qn:
        return df
    return df[
        df["business_name"].str.lower().str.contains(qn, na=False) |
        df["category"].str.lower().str.contains(qn, na=False) |
        df["service"].str.lower().str.contains(qn, na=False) |
        df["notes"].str.lower().str.contains(qn, na=False) |
        df["computed_keywords"].str.lower().str.contains(qn, na=False)
    ]

def main():
    st.title("HCR Providers — Read-Only")

    engine = build_engine()
    version = _get_data_version(engine)
    df = load_df(engine, version)

    q = st.text_input("Search", key="q", placeholder="e.g., roofer, manicure, Bosch, irrigation…")
    dfv = _filter(df, q)

    st.caption(f"{len(dfv)} of {len(df)} providers")
    if dfv.empty:
        st.info("No matches. Try fewer words.")
        return

    # Display table (format only visible)
    dfv_disp = dfv.copy()
    dfv_disp["phone"] = dfv_disp["phone"].map(_fmt_phone)
    cols = ["business_name","category","service","phone","website","notes"]
    cols = [c for c in cols if c in dfv_disp.columns]
    st.dataframe(dfv_disp[cols], use_container_width=True, hide_index=True)

    # Export controls
    c1, c2 = st.columns([1,1])
    with c1:
        csv = dfv_disp.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name="providers.csv", mime="text/csv")
    with c2:
        try:
            import io
            from pandas import ExcelWriter
            import xlsxwriter  # ensure in requirements
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
                dfv_disp.to_excel(w, index=False, sheet_name="Providers")
            st.download_button("Download XLSX", data=bio.getvalue(), file_name="providers.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            pass

    if SHOW_STATUS:
        st.sidebar.info(f"ver {APP_VER} | rows {len(df)} | data_version {version}")

    if SHOW_DIAGS:
        with st.expander("Diagnostics"):
            st.write(df.head(3))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Read-Only app crashed. See logs for details.")
        st.exception(e)
