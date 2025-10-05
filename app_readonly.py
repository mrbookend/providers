from __future__ import annotations

import os
import re
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# -----------------------------
# Layout & page config (full width, no left margin; wrap text)
# -----------------------------
PAGE_TITLE = st.secrets.get("page_title", "Vendors Directory") if hasattr(st, "secrets") else "Vendors Directory"
SIDEBAR_STATE = st.secrets.get("sidebar_state", "collapsed") if hasattr(st, "secrets") else "collapsed"

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

st.markdown(
    """
    <style>
      /* full width, remove side margins/padding */
      .block-container {
        margin-left: 0; margin-right: 0;
        padding-left: 0; padding-right: 0;
        max-width: 100%;
      }
      /* readonly table: wrap text at words; allow rows to auto-grow */
      div[data-testid="stDataFrame"] table {
        white-space: normal;            /* allow wrapping */
        word-break: normal;             /* break at word boundaries */
        overflow-wrap: break-word;      /* but still break very long tokens (urls) */
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# DB (Embedded Replica → sync to Turso; fallback to local file)
# -----------------------------
def build_engine() -> tuple[Engine, dict]:
    """Use Embedded Replica for Turso (syncs to remote), else fallback to local."""
    info: dict = {}

    url   = (st.secrets.get("TURSO_DATABASE_URL") or os.getenv("TURSO_DATABASE_URL") or "").strip()
    token = (st.secrets.get("TURSO_AUTH_TOKEN")   or os.getenv("TURSO_AUTH_TOKEN")   or "").strip()

    if not url:
        engine = create_engine("sqlite:///vendors.db", pool_pre_ping=True)
        info.update({
            "using_remote": False,
            "sqlalchemy_url": "sqlite:///vendors.db",
            "dialect": engine.dialect.name,
            "driver": getattr(engine.dialect, "driver", ""),
        })
        return engine, info

    try:
        engine = create_engine(
            "sqlite+libsql:///vendors-embedded.db",
            connect_args={
                "auth_token": token,   # correct key for libsql-client 0.3.x
                "sync_url": url,       # e.g. libsql://vendors-prod-...turso.io
            },
            pool_pre_ping=True,
        )
        with engine.connect() as c:
            c.execute(sql_text("SELECT 1"))
        info.update({
            "using_remote": True,
            "strategy": "embedded_replica",
            "sqlalchemy_url": "sqlite+libsql:///vendors-embedded.db",
            "dialect": engine.dialect.name,
            "driver": getattr(engine.dialect, "driver", ""),
        })
        return engine, info
    except Exception as e:
        info["remote_error"] = str(e)
        engine = create_engine("sqlite:///vendors.db", pool_pre_ping=True)
        info.update({
            "using_remote": False,
            "sqlalchemy_url": "sqlite:///vendors.db",
            "dialect": engine.dialect.name,
            "driver": getattr(engine.dialect, "driver", ""),
        })
        return engine, info

def table_exists(engine: Engine, name: str) -> bool:
    try:
        with engine.connect() as conn:
            row = conn.execute(
                sql_text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
                {"n": name},
            ).fetchone()
        return bool(row)
    except Exception:
        return False


# -----------------------------
# Data loading
# -----------------------------
def load_df(engine: Engine) -> pd.DataFrame:
    # If initial sync hasn't brought the schema yet, render a friendly placeholder.
    if not table_exists(engine, "vendors"):
        st.info("Connecting to the database for the first time… initial sync may take a few seconds. Refresh to see data once available.")
        expected = ["id","category","service","business_name","contact_name","phone","address","website","notes","keywords","created_at","updated_at","updated_by"]
        return pd.DataFrame(columns=expected)

    # Default sort = Business Name (as requested)
    with engine.begin() as conn:
        df = pd.read_sql(sql_text("SELECT * FROM vendors ORDER BY lower(business_name)"), conn)

    expected = ["id","category","service","business_name","contact_name","phone","address","website","notes","keywords","created_at","updated_at","updated_by"]
    for col in expected:
        if col not in df.columns:
            df[col] = ""
    df = df.fillna("")
    return df

# -----------------------------
# UI
# -----------------------------
engine, engine_info = build_engine()
st.title("Vendors — Read-only")

# Optional help text (from secrets)
help_title = st.secrets.get("READONLY_HELP_TITLE", "Provider Help / Tips")
help_md = st.secrets.get("READONLY_HELP_MD", "")
with st.expander(help_title, expanded=False):
    st.markdown(help_md or "- Type part of a word (e.g., `plumb`).\n- Click a column header to sort.\n- Links open in a new tab.")

st.caption("Global search (case-insensitive; partial words ok). No per-column filters. Read-only.")
q = st.text_input("Search", placeholder="e.g., plumb returns any record with 'plumb' anywhere")

raw = load_df(engine)

# Global contains filter across main columns
if q:
    qq = re.escape(q)
    m = (
        raw["category"].astype(str).str.contains(qq, case=False, na=False) |
        raw["service"].astype(str).str.contains(qq, case=False, na=False) |
        raw["business_name"].astype(str).str.contains(qq, case=False, na=False) |
        raw["contact_name"].astype(str).str.contains(qq, case=False, na=False) |
        raw["phone"].astype(str).str.contains(qq, case=False, na=False) |
        raw["address"].astype(str).str.contains(qq, case=False, na=False) |
        raw["website"].astype(str).str.contains(qq, case=False, na=False) |
        raw["notes"].astype(str).str.contains(qq, case=False, na=False) |
        raw["keywords"].astype(str).str.contains(qq, case=False, na=False)
    )
    df = raw[m]
else:
    df = raw

# Choose visible columns & friendly labels
cols = ["category","business_name","contact_name","phone","address","website","notes","service","keywords","id"]
view = df[cols]

st.data_editor(
    view,
    use_container_width=True,
    disabled=True,           # read-only
    hide_index=True,
    column_config={
        "category": st.column_config.TextColumn("Category"),
        "business_name": st.column_config.TextColumn("Business Name"),
        "contact_name": st.column_config.TextColumn("Contact Name"),
        "phone": st.column_config.TextColumn("Phone"),
        "address": st.column_config.TextColumn("Address"),
        "website": st.column_config.LinkColumn("Website"),
        "notes": st.column_config.TextColumn("Notes"),
        "service": st.column_config.TextColumn("Service"),
        "keywords": st.column_config.TextColumn("Keywords"),
        "id": st.column_config.NumberColumn("ID", disabled=True),
    },
)

st.download_button(
    "Download providers.csv",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name="providers.csv",
    mime="text/csv",
)

# Optional debug (opt-in via Secrets)
if st.secrets.get("ADMIN_DEBUG", "false").lower() in ("1","true","yes"):
    st.divider()
    st.subheader("Status & Secrets (debug)")
    st.json(engine_info)
    with engine.begin() as conn:
        vendors_cols = conn.execute(sql_text("PRAGMA table_info(vendors)")).fetchall()
        counts = {"vendors": conn.execute(sql_text("SELECT COUNT(*) FROM vendors")).scalar() or 0}
    st.write("DB Probe")
    st.json({
        "vendors_columns": [c[1] for c in vendors_cols],
        "counts": counts,
    })
