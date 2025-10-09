# app_readonly.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import io
import re
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# ---- register libsql dialect (must be AFTER "import streamlit as st") ----
try:
    import sqlalchemy_libsql  # noqa: F401  (ensures 'sqlite+libsql' dialect is registered)
except Exception:
    pass
# -------------------------------------------------------------------------


# =============================
# Helpers
# =============================
def _as_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Prefer Streamlit secrets, fallback to environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def _format_phone(val: str | None) -> str:
    s = re.sub(r"\D", "", str(val or ""))
    if len(s) == 10:
        return f"({s[0:3]}) {s[3:6]}-{s[6:10]}"
    return (val or "").strip()


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Returns an .xlsx file (as bytes) using xlsxwriter if present, else openpyxl.
    """
    out = io.BytesIO()
    engine = None
    # prefer xlsxwriter if available
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            engine = "openpyxl"
        except Exception:
            # last resort: tell user in UI later
            raise RuntimeError("Missing Excel writer. Please add 'xlsxwriter' or 'openpyxl' to requirements.txt.")

    with pd.ExcelWriter(out, engine=engine) as writer:
        df.to_excel(writer, index=False, sheet_name="providers")
    out.seek(0)
    return out.read()


# =============================
# Page config & CSS
# =============================
PAGE_TITLE = _get_secret("page_title", "Providers — Read-only") or "Providers — Read-only"
SIDEBAR_STATE = _get_secret("sidebar_state", "collapsed") or "collapsed"
st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

LEFT_PAD_PX = int(_get_secret("page_left_padding_px", "20") or "20")
MAX_WIDTH_PX = int(_get_secret("page_max_width_px", "2300") or "2300")

st.markdown(
    f"""
    <style>
      /* Main container padding and max width */
      [data-testid="stAppViewContainer"] .main .block-container {{
        padding-left: {LEFT_PAD_PX}px !important;
        padding-right: 0 !important;
        max-width: {MAX_WIDTH_PX}px !important;
      }}
      /* Keep data table full width; avoid shrinking due to the narrow search column */
      div[data-testid="stDataFrame"] table {{ white-space: nowrap; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Cosmetic column labels/widths from secrets (optional)
READONLY_COLUMN_LABELS = (
    _get_secret("READONLY_COLUMN_LABELS") if isinstance(_get_secret("READONLY_COLUMN_LABELS"), dict) else None
)
# Fallback: allow a simple mapping defined in secrets.toml-like dict style flattened by Streamlit
# If you provided a dict in Secrets, st.secrets already gives a proper dict. If not, ignore.

COLUMN_WIDTHS = _get_secret("COLUMN_WIDTHS_PX_READONLY")
if not isinstance(COLUMN_WIDTHS, dict):
    COLUMN_WIDTHS = {
        "provider": 220,
        "category": 160,
        "service": 160,
        "contact_name": 180,
        "phone": 140,
        "address": 260,
        "website": 220,
        "notes": 420,
    }

ENABLE_DEBUG = _as_bool(_get_secret("READONLY_MAINTENANCE_ENABLE", "0"), False)


# =============================
# Engine (sanitized + validated)
# =============================
def build_engine() -> Tuple[Engine, Dict]:
    """
    Prefer Turso/libsql embedded replica (read-only behavior here),
    else fall back to local vendors.db if FORCE_LOCAL=1.
    """
    info: Dict = {}
    url_raw = (_get_secret("TURSO_DATABASE_URL", "") or "").strip()
    token = (_get_secret("TURSO_AUTH_TOKEN", "") or "").strip()

    force_local = _as_bool(_get_secret("FORCE_LOCAL", "0"), False)

    if not url_raw or force_local:
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

    # Sanitize sync_url: must be libsql://host (no query params)
    if url_raw.startswith("sqlite+libsql://"):
        host = url_raw.split("://", 1)[1].split("?", 1)[0]
        sync_url = f"libsql://{host}"
    else:
        sync_url = url_raw.split("?", 1)[0]

    try:
        eng = create_engine(
            "sqlite+libsql:///vendors-embedded.db",
            connect_args={
                "auth_token": token,
                "sync_url": sync_url,
            },
            pool_pre_ping=True,
        )
        # lightweight validation
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
        if force_local:
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

        # Read-only app: do not fall back silently unless FORCE_LOCAL=1
        st.error("Remote database is unavailable and FORCE_LOCAL is not enabled.")
        raise


# =============================
# Data loading (read-only)
# =============================
@st.cache_data(show_spinner=False)
def load_df(engine: Engine) -> pd.DataFrame:
    # Read all vendors ordered by name (lowercased for stable order)
    with engine.begin() as conn:
        df = pd.read_sql(sql_text("SELECT * FROM vendors ORDER BY lower(business_name)"), conn)

    # Ensure expected columns exist (for robustness)
    for col in [
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

    # Display-friendly phone
    df["phone_fmt"] = df["phone"].apply(_format_phone)

    # Prebuild a lowercase blob for fast, non-regex contains()
    if "_blob" not in df.columns:
        parts: List[pd.Series] = []
        for c in ["business_name", "category", "service", "contact_name", "phone", "address", "website", "notes", "keywords"]:
            if c in df.columns:
                parts.append(df[c].astype(str))
        df["_blob"] = pd.concat(parts, axis=1).agg(" ".join, axis=1).str.lower() if parts else ""

    return df


# =============================
# UI
# =============================
engine, engine_info = build_engine()
# NOTE: Read-only app — do NOT mutate schema or data. No ensure_schema() call.

st.title(PAGE_TITLE)

# Load once with caching
df = load_df(engine)

# --- Top bar: 25% width search input, table remains full width ---
left, right = st.columns([0.25, 0.75], vertical_alignment="bottom")

with left:
    q = st.text_input(
        "Search",  # non-empty label avoids Streamlit warnings
        placeholder="Search providers… (press Enter)",
        label_visibility="collapsed",
        key="q",
    )

# No refresh button in read-only app; right column intentionally left blank
with right:
    st.write("")

# --- Fast local filtering using prebuilt _blob (no regex) ---
qq = (st.session_state.get("q") or "").strip().lower()
if qq:
    filtered = df[df["_blob"].str.contains(qq, regex=False, na=False)]
else:
    filtered = df

# --- Columns to display (friendly phone) ---
view_cols = [
    "category",
    "service",
    "business_name",
    "contact_name",
    "phone_fmt",
    "address",
    "website",
    "notes",
    "keywords",
]
present = [c for c in view_cols if c in filtered.columns]
vdf = filtered[present].rename(columns={"phone_fmt": "phone"})
vdf = vdf.reset_index(drop=True)

# Column labels (optional)
rename_map = {}
if isinstance(READONLY_COLUMN_LABELS, dict):
    # e.g., {"business_name": "Provider"} or {"provider": "Provider"} depending on your secrets
    rename_map = READONLY_COLUMN_LABELS
    vdf = vdf.rename(columns=rename_map)

# Render table (read-only)
st.dataframe(
    vdf,
    use_container_width=True,
    hide_index=True,
)

# --- Downloads ---
csv_bytes = vdf.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered view (CSV)",
    data=csv_bytes,
    file_name="providers.csv",
    mime="text/csv",
)

# Excel download (xlsx)
try:
    xlsx_bytes = _to_excel_bytes(vdf)
    st.download_button(
        "Download filtered view (Excel)",
        data=xlsx_bytes,
        file_name="providers.xlsx",
        mime=(
            # generic Excel MIME (works broadly)
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
    )
except RuntimeError as e:
    # If missing writer engine, show a soft note
    st.info(f"Excel export unavailable: {e}")

# --- Optional debug panel ---
if ENABLE_DEBUG:
    st.divider()
    btn_label = "Show debug" if not st.session_state.get("show_debug") else "Hide debug"
    if st.button(btn_label):
        st.session_state["show_debug"] = not st.session_state.get("show_debug", False)
        st.rerun()

    if st.session_state.get("show_debug"):
        st.subheader("Status & Secrets (debug)")
        st.json(
            {
                "using_remote": engine_info.get("using_remote"),
                "strategy": engine_info.get("strategy"),
                "sqlalchemy_url": engine_info.get("sqlalchemy_url"),
                "dialect": engine_info.get("dialect"),
                "driver": engine_info.get("driver"),
                "sync_url": engine_info.get("sync_url"),
                "remote_error": engine_info.get("remote_error"),
            }
        )
        with engine.begin() as conn:
            idx_rows = conn.execute(sql_text("PRAGMA index_list(vendors)")).fetchall()
            vendors_indexes = [
                {"seq": r[0], "name": r[1], "unique": bool(r[2]), "origin": r[3], "partial": bool(r[4])} for r in idx_rows
            ]
            counts = {
                "vendors": conn.execute(sql_text("SELECT COUNT(*) FROM vendors")).scalar() or 0,
                "categories": conn.execute(sql_text("SELECT COUNT(*) FROM categories")).scalar() or 0,
                "services": conn.execute(sql_text("SELECT COUNT(*) FROM services")).scalar() or 0,
            }
        st.subheader("DB Probe")
        st.json({"counts": counts, "vendors_indexes": vendors_indexes})
