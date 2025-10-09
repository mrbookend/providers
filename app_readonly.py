# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from io import BytesIO
from typing import Dict, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# Ensure the libsql dialect (must be AFTER streamlit import)
try:
    import sqlalchemy_libsql  # noqa: F401
except Exception:
    pass

# ---------- Optional Excel engine availability ----------
try:
    import xlsxwriter  # noqa: F401
    _xlsx_available = True
except Exception:
    _xlsx_available = False


# =============================
# Helpers
# =============================
def _as_bool(v, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _get_secret(name: str, default: str | None = None) -> str | None:
    """Prefer Streamlit secrets, fallback to environment."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def _to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="providers")
    out.seek(0)
    return out.read()


# =============================
# Page config & CSS
# =============================
PAGE_TITLE = _get_secret("page_title", "Providers — Read-only") or "Providers — Read-only"
SIDEBAR_STATE = _get_secret("sidebar_state", "collapsed") or "collapsed"
LEFT_PAD_PX = int(_get_secret("page_left_padding_px", "20") or "20")
MAX_WIDTH_PX = int(_get_secret("page_max_width_px", "2300") or "2300")

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

st.markdown(
    f"""
    <style>
      .app-container-max {{
        max-width:{MAX_WIDTH_PX}px;
        margin: 0 auto;
      }}
      [data-testid="stAppViewContainer"] .main .block-container {{
        padding-left: {LEFT_PAD_PX}px !important;
        padding-right: 0 !important;
      }}
      /* Prevent table text wrapping for compact view */
      div[data-testid="stDataFrame"] table {{ white-space: nowrap; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-container-max">', unsafe_allow_html=True)


# =============================
# Engine builder (sanitized)
# =============================
def build_engine() -> Tuple[Engine, Dict]:
    """
    Prefer Turso embedded replica (sqlite+libsql) if TURSO_DATABASE_URL is provided.
    Sanitize URL to libsql:// form without query params. Validate connectivity.
    Fallback to local sqlite only if FORCE_LOCAL=1 (string truthy).
    """
    info: Dict = {}

    url = (_get_secret("TURSO_DATABASE_URL", "") or "").strip()
    token = (_get_secret("TURSO_AUTH_TOKEN", "") or "").strip()
    force_local = _as_bool(_get_secret("FORCE_LOCAL", "0"), False)

    if not url:
        # No remote at all → local only
        eng = create_engine("sqlite:///vendors.db", pool_pre_ping=True)
        info.update(
            {
                "using_remote": False,
                "strategy": "local_sqlite",
                "sqlalchemy_url": "sqlite:///vendors.db",
                "dialect": eng.dialect.name,
                "driver": getattr(eng.dialect, "driver", ""),
            }
        )
        return eng, info

    # Sanitize to libsql://<host> (strip query params, accept sqlite+libsql:// forms)
    try:
        raw = url
        if raw.startswith("sqlite+libsql://"):
            host = raw.split("://", 1)[1].split("?", 1)[0]
            sync_url = f"libsql://{host}"
        else:
            # Already libsql://... → strip any query string
            if "://" in raw:
                scheme, rest = raw.split("://", 1)
                if scheme != "libsql":
                    raise ValueError("TURSO_DATABASE_URL must use libsql:// scheme for embedded mode.")
                rest = rest.split("?", 1)[0]
                sync_url = f"libsql://{rest}"
            else:
                raise ValueError("Invalid TURSO_DATABASE_URL format.")
        if not token:
            raise ValueError("TURSO_AUTH_TOKEN is missing for remote database.")
    except Exception as e:
        if force_local:
            eng = create_engine("sqlite:///vendors.db", pool_pre_ping=True)
            info.update(
                {
                    "using_remote": False,
                    "strategy": "local_sqlite",
                    "warn": f"Remote URL invalid: {e}. Using local due to FORCE_LOCAL=1.",
                    "sqlalchemy_url": "sqlite:///vendors.db",
                }
            )
            return eng, info
        st.error(f"Remote DB configuration invalid: {e}")
        raise

    # Attempt embedded replica (file-backed)
    try:
        eng = create_engine(
            "sqlite+libsql:///vendors-embedded.db",
            connect_args={"auth_token": token, "sync_url": sync_url},
            pool_pre_ping=True,
        )
        # Validate connectivity
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
        if force_local:
            eng = create_engine("sqlite:///vendors.db", pool_pre_ping=True)
            info.update(
                {
                    "using_remote": False,
                    "strategy": "local_sqlite",
                    "warn": f"Remote unavailable: {e}. Using local due to FORCE_LOCAL=1.",
                    "sqlalchemy_url": "sqlite:///vendors.db",
                }
            )
            return eng, info
        st.error(f"Remote DB unavailable and FORCE_LOCAL is 0. Error: {e}")
        raise


def ensure_schema(engine: Engine) -> None:
    """Read-only app, but creating tables if absent is harmless and prevents crashes."""
    stmts = [
        """
        CREATE TABLE IF NOT EXISTS vendors (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          category TEXT NOT NULL,
          service TEXT,
          business_name TEXT NOT NULL,
          contact_name TEXT,
          phone TEXT,
          address TEXT,
          website TEXT,
          notes TEXT,
          keywords TEXT,
          created_at TEXT,
          updated_at TEXT,
          updated_by TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_vendors_cat ON vendors(category)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_bus ON vendors(business_name)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_kw  ON vendors(keywords)",
    ]
    with engine.begin() as conn:
        for s in stmts:
            conn.execute(sql_text(s))


@st.cache_data(show_spinner=False)
def load_df(engine: Engine) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(sql_text("SELECT * FROM vendors ORDER BY lower(business_name)"), conn)

    # Friendly phone (don't change stored digits)
    def _format_phone(val: str | None) -> str:
        s = re.sub(r"\D", "", str(val or ""))
        if len(s) == 10:
            return f"({s[0:3]}) {s[3:6]}-{s[6:10]}"
        return (val or "").strip()

    if "phone" in df.columns:
        df["phone_fmt"] = df["phone"].apply(_format_phone)
    else:
        df["phone_fmt"] = ""

    # Build a lowercase search blob once (guarded)
    if "_blob" not in df.columns:
        parts = []
        for col in ["business_name", "category", "service", "contact_name", "phone", "address", "website", "notes", "keywords"]:
            if col in df.columns:
                parts.append(df[col].astype(str))
        df["_blob"] = (pd.concat(parts, axis=1).agg(" ".join, axis=1).str.lower()) if parts else ""

    return df


# =============================
# App UI
# =============================
engine, engine_info = build_engine()
ensure_schema(engine)

st.title(PAGE_TITLE)

# Optional maintenance/debug toggle (no refresh button here)
enable_debug = _as_bool(_get_secret("READONLY_MAINTENANCE_ENABLE", "0"), False)
if enable_debug and "show_debug" not in st.session_state:
    st.session_state["show_debug"] = False

# Top row: left (search), right (optional debug toggle)
top_l, top_r = st.columns([1, 1])

with top_l:
    # Non-empty label avoids repeated Streamlit warnings
    q = st.text_input(
        "Search",
        placeholder="Search providers… (press Enter)",
        label_visibility="collapsed",
        key="q",
    )

with top_r:
    if enable_debug:
        btn_label = "Show debug" if not st.session_state["show_debug"] else "Hide debug"
        if st.button(btn_label):
            st.session_state["show_debug"] = not st.session_state["show_debug"]
            st.rerun()

# Load data
df = load_df(engine)

# Fast local search (no regex): uses prebuilt _blob
qq = (st.session_state.get("q") or "").strip().lower()
if qq:
    filtered = df[df["_blob"].str.contains(qq, regex=False, na=False)]
else:
    filtered = df

# Columns to show (hide id & keywords, show formatted phone)
view_cols = [
    "category",
    "service",
    "business_name",
    "contact_name",
    "phone_fmt",
    "address",
    "website",
    "notes",
]
vdf = filtered.copy()
vdf = vdf.reindex(columns=[c for c in view_cols if c in vdf.columns]).rename(columns={"business_name": "provider", "phone_fmt": "phone"})

# Cosmetic renames via secrets (optional)
# Example in Secrets: READONLY_COLUMN_LABELS = { provider = "Provider" }
label_map = {}
try:
    # st.secrets requires TOML-like mapping; convert to dict if present
    if "READONLY_COLUMN_LABELS" in st.secrets:
        # Streamlit returns MappingProxy, safe to cast
        label_map = dict(st.secrets["READONLY_COLUMN_LABELS"])
except Exception:
    pass

if label_map:
    vdf = vdf.rename(columns=label_map)

# Widths (optional): COLUMN_WIDTHS_PX_READONLY in Secrets
col_config = {}
try:
    widths = st.secrets.get("COLUMN_WIDTHS_PX_READONLY", None)
    if widths:
        # Build column configs with widths where provided
        for col in vdf.columns:
            w = widths.get(col) if isinstance(widths, dict) else None
            if w:
                col_config[col] = st.column_config.TextColumn(col, width=int(w))
except Exception:
    pass

st.data_editor(
    vdf,
    use_container_width=True,  # full width table
    hide_index=True,
    disabled=True,
    column_config=col_config if col_config else None,
)

# ---- Downloads ----
st.download_button(
    "Download filtered view (CSV)",
    data=vdf.to_csv(index=False).encode("utf-8"),
    file_name="providers.csv",
    mime="text/csv",
)
if _xlsx_available:
    st.download_button(
        "Download filtered view (Excel)",
        data=_to_xlsx_bytes(vdf),
        file_name="providers.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.caption("ⓘ Excel export unavailable (install XlsxWriter to enable).")

# ---- Debug panel (optional) ----
if enable_debug and st.session_state.get("show_debug"):
    st.divider()
    st.subheader("Status & Secrets (debug)")
    st.json(engine_info)

    with engine.begin() as conn:
        vendors_cols = conn.execute(sql_text("PRAGMA table_info(vendors)")).fetchall()
        idx_rows = conn.execute(sql_text("PRAGMA index_list(vendors)")).fetchall()
        counts = {"vendors": conn.execute(sql_text("SELECT COUNT(*) FROM vendors")).scalar() or 0}

    vendors_indexes = [
        {"seq": r[0], "name": r[1], "unique": bool(r[2]), "origin": r[3], "partial": bool(r[4])} for r in idx_rows
    ]
    st.json(
        {
            "vendors_columns": [c[1] for c in vendors_cols],
            "counts": counts,
            "vendors_indexes": vendors_indexes,
        }
    )

st.markdown("</div>", unsafe_allow_html=True)
