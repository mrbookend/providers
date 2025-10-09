from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# Ensure libsql dialect is registered
try:
    import sqlalchemy_libsql  # noqa: F401
except Exception:
    pass

# =============================
# Page config & style
# =============================
PAGE_TITLE = (
    st.secrets.get("page_title", "Providers (Read-only)")
    if hasattr(st, "secrets")
    else "Providers (Read-only)"
)
SIDEBAR_STATE = (
    st.secrets.get("sidebar_state", "expanded")
    if hasattr(st, "secrets")
    else "expanded"
)
MAX_WIDTH = int(st.secrets.get("page_max_width_px", 2000)) if hasattr(st, "secrets") else 2000
LEFT_PAD_PX = int(st.secrets.get("page_left_padding_px", 20)) if hasattr(st, "secrets") else 20

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)
st.markdown(
    f"""
    <style>
      .block-container {{
        padding-left: {LEFT_PAD_PX}px !important;
        padding-right: 0 !important;
        max-width: {MAX_WIDTH}px !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Engine builder (sanitizer + validator)
# =============================
def _strip_invisible(s: str) -> str:
    # Remove control/format characters that can break URIs
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith(("C",)))

def _clean_secret(s: str | None) -> str:
    s = (s or "")
    # Normalize curly quotes, stray apostrophes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = _strip_invisible(s).strip().strip('"').strip("'")
    return s

def _normalize_sync_url(raw: str | None) -> str:
    r = _clean_secret(raw)
    if not r:
        return ""
    if r.startswith("sqlite+libsql://"):
        host = r.split("://", 1)[1].split("?", 1)[0]
        return f"libsql://{host}"
    if r.startswith(("http://", "https://")):
        host = r.split("://", 1)[1].split("/", 1)[0]
        return f"libsql://{host}"
    return r

@st.cache_resource
def build_engine() -> Tuple[Engine, dict[str, Any]]:
    """
    Build a libsql embedded-replica engine if TURSO_* secrets are present and valid.
    Fallback to local sqlite if FORCE_LOCAL=1 or if remote is invalid/unavailable (with message).
    """
    info: dict[str, Any] = {}

    url_raw = (st.secrets.get("TURSO_DATABASE_URL") or os.getenv("TURSO_DATABASE_URL"))
    token_raw = (st.secrets.get("TURSO_AUTH_TOKEN") or os.getenv("TURSO_AUTH_TOKEN"))

    sync_url = _normalize_sync_url(url_raw)
    auth_token = _clean_secret(token_raw)
    force_local = str(st.secrets.get("FORCE_LOCAL", "")) == "1" or os.getenv("FORCE_LOCAL") == "1"

    def _local(reason: str | None = None) -> Tuple[Engine, dict[str, Any]]:
        eng = create_engine("sqlite:///vendors.db", pool_pre_ping=True)
        info.update({
            "using_remote": False,
            "strategy": "local_sqlite",
            "sqlalchemy_url": "sqlite:///vendors.db",
            "dialect": eng.dialect.name,
            "driver": getattr(eng.dialect, "driver", ""),
        })
        if reason:
            info["remote_error"] = reason
        return eng, info

    # No remote URL -> local (or stop if not allowed)
    if not sync_url:
        if force_local:
            return _local("No TURSO_DATABASE_URL provided")
        st.error("Remote DB URI missing. Provide TURSO_DATABASE_URL or set FORCE_LOCAL=1 to run locally.")
        st.stop()

    # Validate scheme and characters before touching the Rust client
    if not sync_url.startswith("libsql://") or any(ord(c) < 32 for c in sync_url):
        if force_local:
            return _local("Invalid libsql sync_url; using local due to FORCE_LOCAL=1")
        st.error("Remote DB URI is invalid. It must start with libsql:// and contain no hidden characters.")
        st.stop()

    try:
        eng = create_engine(
            "sqlite+libsql:///vendors-embedded.db",
            connect_args={
                "auth_token": auth_token,
                "sync_url": sync_url,
            },
            pool_pre_ping=True,
        )
        # Sanity probe
        with eng.connect() as c:
            c.exec_driver_sql("select 1;")

        info.update({
            "using_remote": True,
            "strategy": "embedded_replica",
            "sqlalchemy_url": "sqlite+libsql:///vendors-embedded.db",
            "dialect": eng.dialect.name,
            "driver": getattr(eng.dialect, "driver", ""),
            "sync_url": sync_url,
        })
        return eng, info

    except Exception as e:
        if force_local:
            return _local(f"{e}")
        st.error("Remote DB unavailable. Set FORCE_LOCAL=1 to load locally while you fix the secret.")
        raise

engine, engine_info = build_engine()

# =============================
# Data access
# =============================
@st.cache_data(ttl=5)
def _cache_key_for_engine(e: Engine) -> str:
    try:
        return e.url.render_as_string(hide_password=True)
    except Exception:
        return "engine"

@st.cache_data(ttl=600, show_spinner=False, hash_funcs={Engine: _cache_key_for_engine})
def load_df(e: Engine) -> pd.DataFrame:
    with e.begin() as conn:
        df = pd.read_sql(
            sql_text("""
                SELECT id, category, service, business_name, contact_name,
                       phone, address, website, notes, keywords,
                       created_at, updated_at, updated_by
                FROM vendors
                ORDER BY business_name COLLATE NOCASE
            """),
            conn,
        )
    return df

# =============================
# Helpers for enrichment
# =============================
def _fmt_phone(v: Any) -> str:
    s = re.sub(r"\D+", "", str(v or ""))
    return f"({s[0:3]}) {s[3:6]}-{s[6:10]}" if len(s) == 10 else str(v or "")

def _sanitize_url(u: Any) -> str:
    u = (str(u or "")).strip()
    if not u:
        return ""
    if not re.match(r"^https?://", u, re.I):
        u = "https://" + u
    return u

def _build_blob(df: pd.DataFrame) -> pd.Series:
    blob_cols = ["category","service","business_name","contact_name","phone","address","website","notes","keywords"]
    cols = [c for c in blob_cols if c in df.columns]
    return (
        df.reindex(columns=cols)
          .fillna("")
          .astype(str)
          .agg(" ".join, axis=1)
          .str.lower()
    )

# =============================
# UI
# =============================
if "show_debug" not in st.session_state:
    st.session_state["show_debug"] = False

top_l, top_r = st.columns([1, 1])
with top_l:
    # Non-empty label; visually collapsed to satisfy accessibility and silence warnings
    q = st.text_input(
        "Search",
        placeholder="Search providers… (press Enter)",
        label_visibility="collapsed",
        key="q",
    )
with top_r:
    if st.button("Refresh data"):
        st.cache_data.clear()

# Load & enrich BEFORE filtering (prevents _blob KeyError)
df = load_df(engine).copy()

if "phone" in df.columns and "phone_fmt" not in df.columns:
    df["phone_fmt"] = df["phone"].map(_fmt_phone)
if "website" in df.columns:
    df["website"] = df["website"].map(_sanitize_url)

# Always provide a _blob column
df["_blob"] = _build_blob(df)

# Now filter safely
_filtered = df
qq = (st.session_state.get("q") or "").strip().lower()
if qq:
    _filtered = _filtered[_filtered["_blob"].str.contains(qq, regex=False, na=False)]

# Build display frame
view_cols = [
    "business_name", "category", "service", "contact_name",
    "phone_fmt", "address", "website", "notes", "keywords",
]
_cols = [c for c in view_cols if c in _filtered.columns]
grid_df = (
    _filtered
    .reindex(columns=_cols)
    .rename(columns={"business_name": "provider", "phone_fmt": "phone"})
)

# Render table
st.dataframe(
    grid_df,
    use_container_width=True,
    column_config={
        "provider": st.column_config.TextColumn("Provider", width=240),
        "category": st.column_config.TextColumn("Category", width=160),
        "service": st.column_config.TextColumn("Service", width=160),
        "contact_name": st.column_config.TextColumn("Contact", width=180),
        "phone": st.column_config.TextColumn("Phone", width=140),
        "address": st.column_config.TextColumn("Address", width=260),
        "website": st.column_config.LinkColumn("Website", max_chars=40),
        "notes": st.column_config.TextColumn("Notes", width=420),
        "keywords": st.column_config.TextColumn("Keywords", width=160),
    },
    height=560,
)

# CSV download (filtered)
st.download_button(
    "Download CSV file of Providers",
    data=grid_df.to_csv(index=False).encode("utf-8"),
    file_name="providers.csv",
    mime="text/csv",
)

# =============================
# Debug panel
# =============================
st.divider()
btn_label = "Show debug" if not st.session_state["show_debug"] else "Hide debug"
if st.button(btn_label):
    st.session_state["show_debug"] = not st.session_state["show_debug"]
    st.rerun()

if st.session_state["show_debug"]:
    st.subheader("Status & Secrets (debug)")
    safe_info = dict(engine_info)
    # mask sync_url tail for safety in screenshots
    if isinstance(safe_info.get("sync_url"), str):
        s = safe_info["sync_url"]
        if len(s) > 20:
            safe_info["sync_url"] = s[:12] + "…•••…" + s[-6:]
    st.json(safe_info)

    try:
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
    except Exception as e:
        st.warning(f"Debug probe failed: {e}")
