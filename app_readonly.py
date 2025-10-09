# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# Register 'sqlite+libsql' dialect (must be AFTER importing streamlit)
try:
    import sqlalchemy_libsql  # noqa: F401
except Exception:
    pass


# -----------------------------
# Page config & light CSS
# -----------------------------
def _get_secret(name: str, default: str | None = None) -> str | None:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


PAGE_TITLE = _get_secret("page_title", "Providers (Read-only)") or "Providers (Read-only)"
SIDEBAR_STATE = _get_secret("sidebar_state", "expanded") or "expanded"
st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

LEFT_PAD_PX = int(_get_secret("page_left_padding_px", "20") or "20")
st.markdown(
    f"""
    <style>
      [data-testid="stAppViewContainer"] .main .block-container {{
        padding-left: {LEFT_PAD_PX}px !important;
        padding-right: 0 !important;
      }}
      div[data-testid="stDataFrame"] table {{ white-space: nowrap; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Engine (embedded replica) + gated local fallback
# -----------------------------
@st.cache_resource
def build_engine() -> tuple[Engine, dict[str, Any]]:
    """
    Connect via Turso libsql embedded replica if configured.
    Fall back to local sqlite:///vendors.db only if FORCE_LOCAL=1.
    """
    info: dict[str, Any] = {}
    url = (_get_secret("TURSO_DATABASE_URL", "") or "").strip()
    token = (_get_secret("TURSO_AUTH_TOKEN", "") or "").strip()

    if not url:
        eng = create_engine("sqlite:///vendors.db", pool_pre_ping=True)
        info.update(
            {
                "using_remote": False,
                "sqlalchemy_url": "sqlite:///vendors.db",
                "dialect": eng.dialect.name,
                "driver": getattr(eng.dialect, "driver", ""),
            }
        )
        return eng, info

    try:
        # Embedded replica requires libsql:// (not sqlite+libsql://...?secure=true)
        if url.startswith("sqlite+libsql://"):
            host = url.split("://", 1)[1].split("?", 1)[0]
            sync_url = f"libsql://{host}"
        else:
            sync_url = url

        eng = create_engine(
            "sqlite+libsql:///vendors-embedded.db",
            connect_args={"auth_token": token, "sync_url": sync_url},
            pool_pre_ping=True,
        )
        with eng.connect() as c:
            c.exec_driver_sql("select 1;")

        info.update(
            {
                "using_remote": True,
                "strategy": "embedded_replica",
                "sqlalchemy_url": "sqlite+libsql:///vendors-embedded.db",
                "dialect": eng.dialect.name,
                "driver": getattr(eng.dialect, "driver", ""),
                "sync_url": sync_url,
            }
        )
        return eng, info

    except Exception as e:
        info["remote_error"] = f"{e}"
        if (_get_secret("FORCE_LOCAL", "0") or "0").strip() == "1":
            eng = create_engine("sqlite:///vendors.db", pool_pre_ping=True)
            info.update(
                {
                    "using_remote": False,
                    "sqlalchemy_url": "sqlite:///vendors.db",
                    "dialect": eng.dialect.name,
                    "driver": getattr(eng.dialect, "driver", ""),
                }
            )
            return eng, info

        st.error("Remote DB unavailable and FORCE_LOCAL is not set. Aborting.")
        raise


engine, engine_info = build_engine()


# -----------------------------
# Helpers + data loader
# -----------------------------
def _cache_key_for_engine(e: Engine) -> str:
    """Stable, secret-safe cache key for the SQLAlchemy Engine."""
    try:
        return e.url.render_as_string(hide_password=True)
    except Exception:
        return "engine"


@st.cache_data(ttl=600, show_spinner=False, hash_funcs={Engine: _cache_key_for_engine})
def load_df(engine: Engine) -> pd.DataFrame:
    # For < 1k rows we can load the full table and filter client-side.
    with engine.begin() as conn:
        df = pd.read_sql(
            sql_text(
                """
                SELECT id, category, service, business_name, contact_name,
                       phone, address, website, notes, keywords,
                       created_at, updated_at, updated_by
                FROM vendors
                ORDER BY business_name COLLATE NOCASE
                """
            ),
            conn,
        )
    # Ensure expected columns exist even if DB schema is older
    for col in [
        "id",
        "category",
        "service",
        "business_name",
        "contact_name",
        "phone",
        "address",
        "website",
        "notes",
        "keywords",
        "created_at",
        "updated_at",
        "updated_by",
    ]:
        if col not in df.columns:
            df[col] = ""
    return df


# -----------------------------
# UI (single page; no tabs)
# -----------------------------
if "show_debug" not in st.session_state:
    st.session_state["show_debug"] = False
if "q" not in st.session_state:
    st.session_state["q"] = ""

# Top controls
left, right = st.columns([1, 1])
with left:
    q = st.text_input("", placeholder="Search providers…", label_visibility="collapsed", key="q")
with right:
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()
    with c2:
        if st.button("Reset search"):
            st.session_state["q"] = ""
            st.rerun()

# Load and enrich once per run
df = load_df(engine)

# ---------- enrich df for display & search ----------
import re as _re

# phone_fmt for display
if "phone_fmt" not in df.columns:
    def _fmt_phone(v):
        s = _re.sub(r"\D+", "", str(v or ""))
        return f"({s[0:3]}) {s[3:6]}-{s[6:10]}" if len(s) == 10 else str(v or "")
    if "phone" in df.columns:
        df = df.assign(phone_fmt=df["phone"].map(_fmt_phone))
    else:
        df["phone_fmt"] = ""

# normalized website for clickable links
if "website_norm" not in df.columns:
    def _san(u):
        u = (u or "").strip()
        if not u:
            return ""
        if not _re.match(r"^https?://", u, _re.I):
            u = "https://" + u
        return u
    if "website" in df.columns:
        df = df.assign(website_norm=df["website"].map(_san))
    else:
        df["website_norm"] = ""

# unified, lowercase search blob across common fields
_blob_cols = ["category", "service", "business_name", "contact_name", "phone", "address", "website", "notes", "keywords"]
def _to_str(x): return "" if x is None else str(x)
df["_blob"] = (
    df.reindex(columns=[c for c in _blob_cols if c in df.columns])
      .applymap(_to_str)
      .agg(" ".join, axis=1)
      .str.lower()
)
# ---------- /enrich ----------

# Filtering (safe, client-side)
_filtered = df
qq = (st.session_state.get("q") or "").strip().lower()
if qq:
    _filtered = _filtered[_filtered["_blob"].str.contains(qq, regex=False, na=False)]

# Columns to show (sanitized link + formatted phone)
view_cols = [
    "business_name", "category", "service", "contact_name",
    "phone_fmt", "address", "website_norm", "notes",
]
cols_present = [c for c in view_cols if c in _filtered.columns]
grid_df = (
    _filtered
    .reindex(columns=cols_present)
    .rename(columns={"business_name": "provider", "phone_fmt": "phone", "website_norm": "website"})
)

# Render fast, virtualized table
st.dataframe(
    grid_df,
    use_container_width=True,
    column_config={
        "provider":     st.column_config.TextColumn("Provider", width=240),
        "category":     st.column_config.TextColumn("Category", width=140),
        "service":      st.column_config.TextColumn("Service", width=160),
        "contact_name": st.column_config.TextColumn("Contact", width=180),
        "phone":        st.column_config.TextColumn("Phone", width=140),
        "address":      st.column_config.TextColumn("Address", width=260),
        "website":      st.column_config.LinkColumn("Website", max_chars=40),
        "notes":        st.column_config.TextColumn("Notes", width=420, max_chars=240),
    },
    height=560,
)

# CSV download (filtered view)
st.download_button(
    "Download CSV file of Providers",
    data=grid_df.to_csv(index=False).encode("utf-8"),
    file_name="providers.csv",
    mime="text/csv",
)

# -----------------------------
# Debug (toggle)
# -----------------------------
st.divider()
btn = "Show debug" if not st.session_state["show_debug"] else "Hide debug"
if st.button(btn):
    st.session_state["show_debug"] = not st.session_state["show_debug"]
    st.rerun()

if st.session_state["show_debug"]:
    st.subheader("Status & Secrets (debug)")
    safe_info = dict(engine_info)
    if isinstance(safe_info.get("sync_url"), str):
        s = safe_info["sync_url"]
        if len(s) > 20:
            safe_info["sync_url"] = s[:10] + "…•••…" + s[-8:]
    st.json(safe_info)

    with engine.begin() as conn:
        vendors_cols = conn.execute(sql_text("PRAGMA table_info(vendors)")).fetchall()
        categories_cols = conn.execute(sql_text("PRAGMA table_info(categories)")).fetchall()
        services_cols = conn.execute(sql_text("PRAGMA table_info(services)")).fetchall()
        counts = {
            "vendors": conn.execute(sql_text("SELECT COUNT(*) FROM vendors")).scalar() or 0,
            "categories": conn.execute(sql_text("SELECT COUNT(*) FROM categories")).scalar() or 0,
            "services": conn.execute(sql_text("SELECT COUNT(*) FROM services")).scalar() or 0,
        }

    st.subheader("DB Probe")
    st.json(
        {
            "vendors_columns": [c[1] for c in vendors_cols],
            "categories_columns": [c[1] for c in categories_cols],
            "services_columns": [c[1] for c in services_cols],
            "counts": counts,
        }
    )
