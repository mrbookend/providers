# app_readonly.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import io
import html
import textwrap
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# Ensure the libsql dialect is registered (sqlite+libsql)
try:
    import sqlalchemy_libsql  # noqa: F401
except Exception:
    pass

# =============================
# Secrets / config helpers
# =============================
def _get_secret(name: str, default: Optional[str | int | dict | bool] = None):
    """Prefer Streamlit secrets; tolerate local/no-secrets contexts."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    # Fallback to env only for simple strings (never for dicts)
    if isinstance(default, (str, type(None))):
        return os.getenv(name, default)
    return default


def _as_bool(v: Optional[str | bool], default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _get_help_md() -> str:
    """Safe help MD: prefer secrets HELP_MD, fall back to a sensible default."""
    md = ""
    try:
        md = (st.secrets.get("HELP_MD", "") or "").strip()
    except Exception:
        md = ""
    if md:
        return md

    # Built-in fallback if no secret present
    return textwrap.dedent(
        """
        # How to use this list
        - Use the global search box below to match any word or partial word across all columns.
        - Click any column header to sort ascending/descending (client-side).
        - Use **Download** at the bottom to export the current table to `providers.csv`.
        - Notes and Address wrap to show more text. Phone is normalized when available.
        """
    ).strip()


# =============================
# Page config
# =============================
PAGE_TITLE = _get_secret("page_title", "HCR Providers — Read-Only")
PAGE_MAX_WIDTH_PX = int(_get_secret("page_max_width_px", 2300))
SIDEBAR_STATE = _get_secret("sidebar_state", "collapsed")

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

st.markdown(
    f"""
    <style>
      .block-container {{
        max-width: {PAGE_MAX_WIDTH_PX}px;
      }}
      /* Make markdown and table text wrap naturally */
      .prov-table td, .prov-table th {{
        white-space: normal;
        word-break: break-word;
        vertical-align: top;
        padding: 6px 8px;
        border-bottom: 1px solid #eee;
      }}
      .prov-table thead th {{
        position: sticky;
        top: 0;
        background: #fff;
        z-index: 2;
        border-bottom: 2px solid #ddd;
      }}
      /* Optional sticky first column support */
      .prov-table td.sticky, .prov-table th.sticky {{
        position: sticky;
        left: 0;
        background: #fff;
        z-index: 3;
        box-shadow: 1px 0 0 #eee;
      }}
      /* Table container allows horizontal scroll if needed */
      .prov-wrap {{
        overflow-x: auto;
      }}
      /* Help button row alignment */
      .help-row {{
        display: flex;
        align-items: center;
        gap: 12px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Column labels & widths (from secrets)
# =============================
# Raw (may be Streamlit's read-only AttrDict)
_readonly_labels_raw = _get_secret("READONLY_COLUMN_LABELS", {}) or {}
_col_widths_raw = _get_secret("COLUMN_WIDTHS_PX_READONLY", {}) or {}

# Normalize to *plain dicts* so we can safely operate on them
READONLY_COLUMN_LABELS: Dict[str, str] = dict(_readonly_labels_raw)
# Coerce width values to int in case TOML strings were used
COLUMN_WIDTHS_PX_READONLY: Dict[str, int] = {str(k): int(v) for k, v in dict(_col_widths_raw).items()}

STICKY_FIRST_COL: bool = _as_bool(_get_secret("READONLY_STICKY_FIRST_COL", False), False)

# Reasonable defaults
DEFAULT_WIDTHS = {
    "id": 40,
    "business_name": 180,
    "category": 175,
    "service": 120,
    "contact_name": 110,
    "phone": 120,
    "address": 220,
    "website": 200,
    "notes": 220,
    "keywords": 90,
}

# Merge defaults (left) with user-specified (right); user values win
COLUMN_WIDTHS_PX_READONLY = {**DEFAULT_WIDTHS, **COLUMN_WIDTHS_PX_READONLY}



# =============================
# Database engine (Turso first; guarded fallback)
# =============================
def build_engine() -> Tuple[Engine, Dict[str, str]]:
    info = {}
    url = _get_secret("TURSO_DATABASE_URL", "") or ""
    token = _get_secret("TURSO_AUTH_TOKEN", "") or ""

    if url and token:
        connect_args = {"auth_token": token}
        engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
        info.update(
            {
                "using_remote": "true",
                "strategy": "remote_only",
                "sqlalchemy_url": url.split("?")[0],
                "dialect": "sqlite",
                "driver": getattr(engine.dialect, "driver", ""),
            }
        )
        return engine, info

    # Fallback (optional; useful for local dev). In prod you may disable this path.
    local_path = os.path.abspath("./vendors.db")
    local_url = f"sqlite:///{local_path}"
    engine = create_engine(local_url, pool_pre_ping=True)
    info.update(
        {
            "using_remote": "false",
            "strategy": "local_fallback",
            "sqlalchemy_url": local_url,
            "dialect": "sqlite",
            "driver": getattr(engine.dialect, "driver", ""),
        }
    )
    return engine, info


# =============================
# Data access
# =============================
VENDOR_COLS = [
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
]


def fetch_vendors(engine: Engine) -> pd.DataFrame:
    sql = "SELECT " + ", ".join(VENDOR_COLS) + " FROM vendors ORDER BY lower(business_name)"
    with engine.begin() as conn:
        df = pd.read_sql(sql_text(sql), conn)
    # Normalize website to clickable markdown if it looks like a URL
    def _mk_link(v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            return ""
        u = v.strip()
        if not (u.startswith("http://") or u.startswith("https://")):
            u = "https://" + u
        # Escape label only; allow href as-is
        label = html.escape(v.strip())
        return f"[{label}]({u})"

    if "website" in df.columns:
        df["website"] = df["website"].fillna("").astype(str).map(_mk_link)
    # Ensure strings (for consistent search)
    for c in df.columns:
        if c not in ("id", "created_at", "updated_at"):  # let id remain int if it is
            df[c] = df[c].fillna("").astype(str)
    return df


# =============================
# Filtering (global search)
# =============================
def apply_global_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = (query or "").strip().lower()
    if not q:
        return df
    # Simple case-insensitive contains across all columns (stringified already)
    mask = pd.Series(False, index=df.index)
    for c in df.columns:
        mask |= df[c].str.lower().str.contains(q, na=False)
    return df[mask]


# =============================
# Rendering
# =============================
def _rename_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Apply label mapping; return df and reverse map (for width lookups)."""
    mapping = {k: READONLY_COLUMN_LABELS.get(k, k.replace("_", " ").title()) for k in df.columns}
    df2 = df.rename(columns=mapping)
    # Reverse map: display -> original
    rev = {v: k for k, v in mapping.items()}
    return df2, rev


def _build_table_html(df: pd.DataFrame, sticky_first: bool) -> str:
    """Render a basic HTML table with widths and wrapping."""
    df_disp, rev = _rename_columns(df)

    # Build header
    headers = []
    for col in df_disp.columns:
        orig = rev[col]
        width = COLUMN_WIDTHS_PX_READONLY.get(orig, 140)
        sticky_cls = "sticky" if sticky_first and (col == df_disp.columns[0]) else ""
        headers.append(f'<th class="{sticky_cls}" style="min-width:{width}px;max-width:{width}px;">{html.escape(col)}</th>')
    thead = "<thead><tr>" + "".join(headers) + "</tr></thead>"

    # Build body
    rows_html: List[str] = []
    first_col_name = df_disp.columns[0] if len(df_disp.columns) else None
    for _, row in df_disp.iterrows():
        tds: List[str] = []
        for col in df_disp.columns:
            orig = rev[col]
            val = row[col]
            width = COLUMN_WIDTHS_PX_READONLY.get(orig, 140)
            # Website column is markdown link; ensure it remains markdown-supported via st.markdown
            cell = str(val) if val is not None else ""
            sticky_cls = "sticky" if (sticky_first and col == first_col_name) else ""
            tds.append(f'<td class="{sticky_cls}" style="min-width:{width}px;max-width:{width}px;">{cell}</td>')
        rows_html.append("<tr>" + "".join(tds) + "</tr>")
    tbody = "<tbody>" + "".join(rows_html) + "</tbody>"

    return f'<div class="prov-wrap"><table class="prov-table">{thead}{tbody}</table></div>'


# =============================
# Main UI
# =============================
def main():
    # Top row: Help button only (no big title banner)
    with st.container():
        cols = st.columns([1, 8, 1])
        with cols[0]:
            open_help = st.button("Help / Tips", type="primary", use_container_width=True)
        with cols[1]:
            st.caption("Use the global search below to match any word or partial word across all columns.")
        # cols[2] left empty for spacing/alignment

    if open_help:
        with st.expander("Provider Help / Tips", expanded=True):
            st.markdown(_get_help_md())

    # Build DB engine
    engine, info = build_engine()
    if info.get("strategy") == "local_fallback":
        st.warning("Turso credentials not found. Running on local SQLite fallback (`./vendors.db`).")

    # Load data
    try:
        df = fetch_vendors(engine)
    except Exception as e:
        st.error(f"Failed to load vendors: {e}")
        return

    # Hide "id" by default in the display
    disp_cols = [c for c in df.columns if c != "id"]
    df = df[disp_cols]

    # Global search
    st.text_input(
        "Search",
        key="q",
        placeholder="e.g., plumb, roofing, 'Inverness', phone digits, etc.",
        help="Case-insensitive, matches partial words across all columns.",
    )
    q = st.session_state.get("q", "")
    filtered = apply_global_search(df, q)

    # Sort client-side by clicking header (not server-side ordering here)
    # Render as lightweight HTML so wrapping + pixel widths are honored.
    html_table = _build_table_html(filtered, sticky_first=STICKY_FIRST_COL)
    st.markdown(html_table, unsafe_allow_html=True)

    # Download CSV (place near the bottom, before debug)
    csv_buf = io.StringIO()
    # Use original column names for CSV, not the labeled versions
    filtered.to_csv(csv_buf, index=False)
    st.download_button(
        "Download providers.csv",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="providers.csv",
        mime="text/csv",
        type="secondary",
        use_container_width=False,
    )

    # Debug section as a toggle button at the very bottom
    st.divider()
    if st.button("Debug (status & secrets keys)", type="secondary"):
        dbg = {
            "DB (resolved)": info,
            "Secrets keys": sorted(list(getattr(st, "secrets", {}).keys())) if hasattr(st, "secrets") else [],
        }
        st.write(dbg)
        # Also show a tiny probe of the DB
        try:
            with engine.begin() as conn:
                cols_v = pd.read_sql(sql_text("PRAGMA table_info(vendors)"), conn)
                cols_c = pd.read_sql(sql_text("PRAGMA table_info(categories)"), conn)
                cols_s = pd.read_sql(sql_text("PRAGMA table_info(services)"), conn)
                counts = {}
                for t in ("vendors", "categories", "services"):
                    c = pd.read_sql(sql_text(f"SELECT COUNT(*) AS n FROM {t}"), conn)["n"].iloc[0]
                    counts[t] = int(c)
            probe = {
                "vendors_columns": list(cols_v["name"]) if "name" in cols_v else [],
                "categories_columns": list(cols_c["name"]) if "name" in cols_c else [],
                "services_columns": list(cols_s["name"]) if "name" in cols_s else [],
                "counts": counts,
            }
            st.write(probe)
        except Exception as e:
            st.write({"db_probe_error": str(e)})


if __name__ == "__main__":
    main()
