# app_readonly.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import io
import re
import html
from typing import Dict, Tuple, List, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# ---- register libsql dialect (must be AFTER "import streamlit as st") ----
try:
    import sqlalchemy_libsql  # noqa: F401
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
    """Returns an .xlsx file (as bytes) using xlsxwriter if present, else openpyxl."""
    out = io.BytesIO()
    engine = None
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            engine = "openpyxl"
        except Exception:
            raise RuntimeError("Missing Excel writer. Please add 'xlsxwriter' or 'openpyxl' to requirements.txt.")

    with pd.ExcelWriter(out, engine=engine) as writer:
        df.to_excel(writer, index=False, sheet_name="providers")
    out.seek(0)
    return out.read()


# =============================
# Page config & CSS
# =============================
PAGE_TITLE = _get_secret("page_title", "HCR Providers (Read-Only)") or "HCR Providers (Read-Only)"
SIDEBAR_STATE = _get_secret("sidebar_state", "collapsed") or "collapsed"
st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)
st.caption(f"Streamlit {st.__version__}")  # temporary; remove later

LEFT_PAD_PX = int(_get_secret("page_left_padding_px", "20") or "20")
MAX_WIDTH_PX = int(_get_secret("page_max_width_px", "2300") or "2300")


# Optional column labels (cosmetic). We’ll also force business_name -> providers below.
READONLY_COLUMN_LABELS = _get_secret("READONLY_COLUMN_LABELS")
if not isinstance(READONLY_COLUMN_LABELS, dict):
    READONLY_COLUMN_LABELS = {}

# Optional per-column pixel widths for the HTML table
COLUMN_WIDTHS = _get_secret("COLUMN_WIDTHS_PX_READONLY")
if not isinstance(COLUMN_WIDTHS, dict):
    COLUMN_WIDTHS = {
        "providers": 220,     # after rename
        "business_name": 220, # keep both keys safe
        "category": 160,
        "service": 160,
        "contact_name": 180,
        "phone": 140,
        "address": 260,
        "website": 220,
        "notes": 420,
        "keywords": 120,
    }

ENABLE_DEBUG = _as_bool(_get_secret("READONLY_MAINTENANCE_ENABLE", "0"), False)

HELP_MD = _get_secret(
    "HELP_MD",
    """# HCR Providers (Read-Only)

**How to use this list**
- Use the search box to find providers by any word fragment (e.g., typing `plumb` matches “Plumbing”, “Plumber”, etc.).
- Click the CSV/Excel buttons to download exactly what you’re viewing.
- Phone is formatted for readability; original data remains unchanged.
- Data are read-only here; changes happen in the Admin app.
"""
)

# Global CSS: layout, modal width, and HTML-table wrapping rules
st.markdown(
    f"""
    <style>
      /* Page padding & max width */
      [data-testid="stAppViewContainer"] .main .block-container {{
        padding-left: {LEFT_PAD_PX}px !important;
        padding-right: 0 !important;
        max-width: {MAX_WIDTH_PX}px !important;
      }}

      /* Make modal close to full width */
      div[data-testid="stModal"] > div {{
        width: min(95vw, {MAX_WIDTH_PX}px) !important;
        max-width: none !important;
      }}

      /* HTML table defaults: no wrap by default; we will enable on chosen cols */
      table.providers-html {{
        width: 100%;
        table-layout: fixed;
        border-collapse: collapse;
        font-size: 0.95rem;
      }}
      table.providers-html thead th {{
        text-align: left;
        border-bottom: 1px solid #ddd;
        padding: 6px 8px;
      }}
      table.providers-html tbody td {{
        padding: 6px 8px;
        vertical-align: top;
        white-space: nowrap;         /* default: keep single-line */
        overflow-wrap: normal;
        word-break: normal;
      }}
      /* Cells that should wrap & grow rows automatically */
      table.providers-html td.wrap {{
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
      }}
      /* Subtle zebra stripes */
      table.providers-html tbody tr:nth-child(odd) {{
        background: rgba(0,0,0,0.02);
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


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

        st.error("Remote database is unavailable and FORCE_LOCAL is not enabled.")
        raise


# =============================
# Data loading (read-only, cached)
# =============================
@st.cache_data(show_spinner=False, ttl=120)
def load_df() -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(sql_text("SELECT * FROM vendors ORDER BY lower(business_name)"), conn)

    # Ensure expected columns exist
    for col in [
        "category","service","business_name","contact_name","phone","address",
        "website","notes","keywords","created_at","updated_at","updated_by",
    ]:
        if col not in df.columns:
            df[col] = ""

    df["phone_fmt"] = df["phone"].apply(_format_phone)

    # Prebuild a lowercase blob for fast contains()
    if "_blob" not in df.columns:
        parts: List[pd.Series] = []
        for c in ["business_name","category","service","contact_name","phone","address","website","notes","keywords"]:
            if c in df.columns:
                parts.append(df[c].astype(str))
        df["_blob"] = pd.concat(parts, axis=1).agg(" ".join, axis=1).str.lower() if parts else ""

    return df


# =============================
# HTML rendering (wrapping + widths)
# =============================
def _width_for(col: str) -> Optional[int]:
    """Look up a width for either the renamed or original column key."""
    return COLUMN_WIDTHS.get(col) or COLUMN_WIDTHS.get({
        "providers": "business_name"
    }.get(col, ""))

def _as_link(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    # Allow plain domains without scheme
    if not re.match(r"^[a-z]+://", u, flags=re.I):
        u = "https://" + u
    safe = html.escape(u)
    return f'<a href="{safe}" target="_blank" rel="noopener noreferrer">{safe}</a>'

def render_html_table(df: pd.DataFrame, wrap_cols: List[str]) -> str:
    cols = list(df.columns)
    # Build table header
    ths = []
    for c in cols:
        label = html.escape(c)
        w = _width_for(c)
        style = f"style='width:{w}px; max-width:{w}px;'" if w else ""
        ths.append(f"<th {style}>{label}</th>")
    thead = "<thead><tr>" + "".join(ths) + "</tr></thead>"

    # Build rows
    trs = []
    for _, row in df.iterrows():
        tds = []
        for c in cols:
            val = row[c]
            if pd.isna(val):
                val = ""
            if c == "website":
                cell_html = _as_link(str(val))
            else:
                cell_html = html.escape(str(val))
            klass = "wrap" if c in wrap_cols else ""
            w = _width_for(c)
            style = f"style='width:{w}px; max-width:{w}px;'" if w else ""
            tds.append(f"<td class='{klass}' {style}>{cell_html}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    tbody = "<tbody>" + "".join(trs) + "</tbody>"

    return f"<table class='providers-html'>{thead}{tbody}</table>"


# =============================
# UI
# =============================
engine, engine_info = build_engine()

# NOTE: per request, NO visible st.title here. (Tab title still set via set_page_config.)

# Load once
df = load_df()

# --- Top controls row: Help button + Search input
left, right = st.columns([0.14, 0.86], vertical_alignment="bottom")

with left:
    if st.button("Help / Instructions", use_container_width=True, key="help_btn"):
        st.session_state["show_help"] = True

with right:
    q = st.text_input(
        "Search",  # keep a non-empty label (collapsed)
        placeholder="Search providers… (press Enter)",
        label_visibility="collapsed",
        key="q",
    )

# Modal (near full width)
if st.session_state.get("show_help"):
    with st.modal("HCR Providers (Read-Only) — Help", key="help_modal"):
        st.markdown(HELP_MD)
        if st.button("Close"):
            st.session_state["show_help"] = False
            st.rerun()

# --- Fast local filtering (case-insensitive, non-regex) ---
qq = (st.session_state.get("q") or "").strip().lower()
filtered = df[df["_blob"].str.contains(qq, regex=False, na=False)] if qq else df

# --- Select & rename columns (business_name -> providers) ---
# --- Select & rename columns (business_name -> providers) ---
# NOTE: 'keywords' is intentionally excluded from display/downloads,
# but it remains part of the search index via '_blob'.
view_cols = [
    "category", "service", "business_name", "contact_name",
    "phone_fmt", "address", "website", "notes",
]
present = [c for c in view_cols if c in filtered.columns]
vdf = filtered[present].rename(columns={"phone_fmt": "phone"})


# Force the business_name label to "providers" regardless of secrets
label_map = {"business_name": "providers"}
if isinstance(READONLY_COLUMN_LABELS, dict):
    label_map.update(READONLY_COLUMN_LABELS)  # allow additional labels via secrets (won’t override our forced one)
vdf = vdf.rename(columns=label_map)

# --- Render HTML table with wrapping on specified columns ---
wrap_cols = ["category", "service", "providers", "contact_name", "address", "website", "notes"]
html_table = render_html_table(vdf, wrap_cols=wrap_cols)
st.markdown(html_table, unsafe_allow_html=True)

# --- Downloads (CSV / Excel) ---
csv_bytes = vdf.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered view (CSV)",
    data=csv_bytes,
    file_name="providers.csv",
    mime="text/csv",
)

try:
    xlsx_bytes = _to_excel_bytes(vdf)
    st.download_button(
        "Download filtered view (Excel)",
        data=xlsx_bytes,
        file_name="providers.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
except RuntimeError as e:
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
