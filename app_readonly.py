# app_readonly.py — HCR Vendors (Read-Only) v3.6
# - AgGrid: NO per-column filters; selected columns wrap + auto-expand rows
# - Columns with wrap+autoHeight: category, service, business_name, contact_name, address, website, notes (+ Website URL)
# - Replaces "View" title with a Help button that opens a wide modal page of instructions (from secrets)
# - Quick filter under Help; CSV at bottom; Debug expander at very bottom
# - Website column shows "Website" link only when URL is valid; adjacent "Website URL" shows full link
# - Reads labels/widths/help from secrets

from __future__ import annotations

import os
from urllib.parse import urlparse
from typing import Dict, List

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.sql import text as sql_text

# ---- AgGrid (optional; app falls back if not installed) ----
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode, ColumnsAutoSizeMode
    _AGGRID_AVAILABLE = True
except Exception:
    _AGGRID_AVAILABLE = False


# -----------------------------
# Secrets / env helpers
# -----------------------------
def _read_secret(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)

def _get_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("1","true","yes","on")
    return bool(default)


# -----------------------------
# Page config / layout
# -----------------------------
def _apply_layout():
    st.set_page_config(
        page_title=_read_secret("page_title", "HCR Vendors (Read-Only)"),
        layout="wide",
        initial_sidebar_state=_read_secret("sidebar_state", "expanded"),
    )
    maxw = _read_secret("page_max_width_px", 2300)
    try:
        maxw = int(maxw)
    except Exception:
        maxw = 2300
    st.markdown(f"<style>.block-container {{ max-width: {maxw}px; }}</style>", unsafe_allow_html=True)

_apply_layout()


# -----------------------------
# Config from secrets
# -----------------------------
LABEL_OVERRIDES: Dict[str, str] = _read_secret("READONLY_COLUMN_LABELS", {}) or {}
COLUMN_WIDTHS: Dict[str, int]   = _read_secret("COLUMN_WIDTHS_PX_READONLY", {}) or {}
STICKY_FIRST: bool              = _get_bool(_read_secret("READONLY_STICKY_FIRST_COL", False), False)

HELP_TITLE: str = (
    _read_secret("READONLY_HELP_TITLE")
    or LABEL_OVERRIDES.get("readonly_help_title")
    or "Providers Help / Tips"
)
HELP_MD: str = (
    _read_secret("READONLY_HELP_MD")
    or LABEL_OVERRIDES.get("readonly_help_md")
    or ""
)
HELP_DEBUG: bool = _get_bool(
    _read_secret("READONLY_HELP_DEBUG"),
    _get_bool(LABEL_OVERRIDES.get("readonly_help_debug"), False),
)

RAW_COLS = ["id","business_name","category","service","contact_name","phone","address","website","notes","keywords"]


# -----------------------------
# Engine (Turso/libSQL with fallback)
# -----------------------------
def _engine():
    url = _read_secret("TURSO_DATABASE_URL", "sqlite+libsql://vendors-prod-mrbookend.aws-us-west-2.turso.io?secure=true")
    token = _read_secret("TURSO_AUTH_TOKEN", None)
    if isinstance(url, str) and url.startswith("sqlite+libsql://"):
        try:
            eng = create_engine(url, connect_args={"auth_token": token} if token else {}, pool_pre_ping=True)
            with eng.connect() as c:
                c.execute(sql_text("SELECT 1"))
            return eng
        except Exception as e:
            st.warning(f"Turso connection failed ({e}). Falling back to local SQLite vendors.db.")
    local = "/mount/src/vendors-readonly-app/vendors.db"
    if not os.path.exists(local):
        local = "vendors.db"
    return create_engine(f"sqlite:///{local}")

eng = _engine()


# -----------------------------
# CSS (fallback table only)
# -----------------------------
def _apply_css(field_order: List[str]):
    rules = []
    for idx, col in enumerate(field_order, start=1):
        px = COLUMN_WIDTHS.get(col)
        if not px: continue
        rules.append(
            f"""
            div[data-testid='stDataFrame'] table thead tr th:nth-child({idx}),
            div[data-testid='stDataFrame'] table tbody tr td:nth-child({idx}) {{
                min-width:{px}px !important; max-width:{px}px !important; width:{px}px !important;
                white-space: normal !important; overflow-wrap:anywhere !important; word-break:break-word !important;
            }}
            """
        )
    sticky = ""
    if STICKY_FIRST and field_order:
        sticky = """
        div[data-testid='stDataFrame'] table thead tr th:nth-child(1),
        div[data-testid='stDataFrame'] table tbody tr td:nth-child(1){
            position:sticky;left:0;z-index:2;background:var(--background-color,white);box-shadow:1px 0 0 rgba(0,0,0,0.06);
        }
        div[data-testid='stDataFrame'] table thead tr th:nth-child(1){z-index:3;}
        """
    st.markdown(
        f"""
        <style>
        div[data-testid='stDataFrame'] table {{ table-layout: fixed !important; }}
        div[data-testid='stDataFrame'] table td, div[data-testid='stDataFrame'] table th {{
            white-space: normal !important; overflow-wrap:anywhere !important; word-break:break-word !important;
        }}
        {''.join(rules)}
        {sticky}
        </style>
        """, unsafe_allow_html=True
    )


# -----------------------------
# URL helper
# -----------------------------
def _normalize_url(u: str) -> str:
    s = (u or "").strip()
    if not s: return ""
    if not s.lower().startswith(("http://","https://")):
        s = "http://" + s
    try:
        p = urlparse(s)
        if p.scheme not in ("http","https"): return ""
        if not p.netloc or "." not in p.netloc: return ""
        return s
    except Exception:
        return ""


# -----------------------------
# Help (modal) + Quick filter
# -----------------------------
# Wide modal with instructions (from secrets)
try:
    # Streamlit >= 1.31
    @st.dialog(HELP_TITLE, width="large")
    def _help_modal():
        if HELP_MD.strip():
            st.markdown(HELP_MD)
            if HELP_DEBUG:
                st.caption("Raw help (debug preview)")
                st.code(HELP_MD, language=None)
        else:
            st.markdown(
                """
                ### Finding Providers
                - Use the **Quick filter** below. Type words or parts of words (e.g., *plumber*, *roof*, ZIP).
                - The filter matches across Provider, Category, Service, Address, Website, Notes, and Keywords.

                ### Sorting & Reading
                - Click a column header to sort.
                - Long text (Address, Notes, Website URL) **wraps** and the row **expands** to fit.

                ### Copying & Links
                - Select text and press **Ctrl/Cmd+C**.
                - Click **Website** to open in a new tab; the **Website URL** column shows the full link.

                ### Column Sizes
                - Columns start at fixed widths for readability; you can resize columns.

                ### Contribute Updates
                - Post recommended changes in the HCR Facebook group.
                """
            )
except Exception:
    _help_modal = None  # Streamlit too old; we’ll fall back

def _help_button_and_filter():
    cols = st.columns([1, 3])
    with cols[0]:
        if st.button("Help / How to Use", type="primary"):
            if _help_modal:
                _help_modal()
            else:
                # Fallback: open an expander if dialogs are unavailable
                with st.expander(HELP_TITLE, expanded=True):
                    st.markdown(HELP_MD or "No help content configured.")
    with cols[1]:
        q = st.text_input("Quick filter (type words or parts of words):", "", placeholder="e.g., plumber roof 78240")
    return q

def _apply_quick_filter(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = (query or "").strip().lower()
    if not q:
        return df
    terms = [t for t in q.split() if t]
    if not terms:
        return df
    cols = [c for c in df.columns if c.lower() in {
        "business_name","provider","category","service","contact_name",
        "phone","address","website","notes","keywords","website url"
    }]
    if not cols:
        return df
    blob = df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    mask = pd.Series(True, index=df.index)
    for t in terms:
        mask &= blob.str.contains(t, na=False)
    return df[mask]


# -----------------------------
# Data access
# -----------------------------
def _fetch_df() -> pd.DataFrame:
    with eng.connect() as c:
        df = pd.read_sql_query(sql_text("SELECT * FROM vendors"), c)
    for col in RAW_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[RAW_COLS]

def _apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    if not LABEL_OVERRIDES:
        return df
    mapping = {k:v for k,v in LABEL_OVERRIDES.items() if k in df.columns and isinstance(v,str) and v}
    return df.rename(columns=mapping)


# -----------------------------
# AgGrid renderer (no per-column filters; selected columns autoHeight)
# -----------------------------
def _aggrid_view(df_show: pd.DataFrame, website_label: str = "website"):
    if df_show.empty:
        st.info("No rows to display.")
        return

    website_key = website_label if website_label in df_show.columns else next(
        (c for c in df_show.columns if c.lower()=="website"), None
    )

    _df = df_show.copy()
    url_col = None
    if website_key:
        raw_guess = "website"
        norm = _df[raw_guess].map(_normalize_url) if raw_guess in _df.columns else _df[website_key].map(_normalize_url)
        url_col = f"{website_key} URL" if website_key.lower() != "website" else "Website URL"
        widx = _df.columns.get_loc(website_key)
        _df.insert(widx + 1, url_col, norm)

    gob = GridOptionsBuilder.from_dataframe(_df)

    # Disable per-column filters; keep sort/resize
    gob.configure_default_column(
        resizable=True,
        sortable=True,
        filter=False,
        wrapText=False,
        autoHeight=False,
    )

    # Widths from secrets (map displayed -> raw if renamed)
    display_to_raw = {disp: raw for raw, disp in LABEL_OVERRIDES.items() if isinstance(disp, str) and disp}
    for col in _df.columns:
        raw_key = display_to_raw.get(col, col)
        px = COLUMN_WIDTHS.get(raw_key)
        if px:
            gob.configure_column(col, width=px)

    # Columns that must wrap + auto expand
    # (category, service, business_name, contact_name, address, website, notes) + dynamic Website URL column
    must_wrap = {"category","service","business_name","contact_name","address","website","notes"}
    # Include displayed names (after label mapping) and dynamic URL col
    wrap_targets = set()
    for col in list(_df.columns):
        low = col.lower()
        raw_guess = display_to_raw.get(col, col).lower()
        if low in must_wrap or raw_guess in must_wrap:
            wrap_targets.add(col)
    if url_col:
        wrap_targets.add(url_col)

    for col in wrap_targets:
        gob.configure_column(
            col,
            wrapText=True,
            autoHeight=True,
            cellStyle={"whiteSpace": "normal", "wordBreak": "break-word", "overflowWrap": "anywhere"},
        )

    # Clickable "Website" label only if URL is valid (url_col exists)
    if website_key and url_col:
        link_renderer = JsCode(f"""
            function(params){{
                const url = params.data && params.data["{url_col}"] ? params.data["{url_col}"] : "";
                if (!url) return "";
                return `<a href="${{url}}" target="_blank" rel="noopener noreferrer">Website</a>`;
            }}
        """)
        gob.configure_column(website_key, cellRenderer=link_renderer)

    grid_options = gob.build()
    grid_options["floatingFilter"] = False
    grid_options["suppressMenuHide"] = True

    AgGrid(
        _df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.NO_UPDATE,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        height=620,  # scrollable grid; stable layout
        theme="streamlit",
    )


# -----------------------------
# View body (help button + quick filter; CSV at bottom)
# -----------------------------
def render_view():
    # Help button (opens modal) and quick filter right under it
    query = _help_button_and_filter()

    df = _fetch_df()
    df_show = _apply_labels(df)
    df_show = _apply_quick_filter(df_show, query)

    if _AGGRID_AVAILABLE:
        _aggrid_view(df_show, website_label=LABEL_OVERRIDES.get("website", "website"))
    else:
        st.warning("`streamlit-aggrid` not installed — showing basic table.")
        _apply_css(df.columns.tolist())
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    # CSV download at the BOTTOM (just ahead of debug)
    st.download_button(
        label="Download Providers (CSV)",
        data=df_show.to_csv(index=False).encode("utf-8"),
        file_name="providers_readonly.csv",
        mime="text/csv",
    )


# -----------------------------
# Status & Secrets (debug) — END (BOTTOM)
# -----------------------------
def render_status_debug():
    with st.expander("Status & Secrets (debug)", expanded=False):
        backend = "libsql" if str(_read_secret("TURSO_DATABASE_URL","")).startswith("sqlite+libsql://") else "sqlite"
        st.write("DB")
        st.code({"backend": backend, "dsn": _read_secret("TURSO_DATABASE_URL",""),
                 "auth": "token_set" if bool(_read_secret("TURSO_AUTH_TOKEN")) else "none"})
        try:
            keys = list(st.secrets.keys())
        except Exception:
            keys = []
        st.write("Secrets keys (present)"); st.code(keys)
        st.write("Help MD:", "present" if bool(HELP_MD) else "(missing or empty)")
        st.write("Sticky first col enabled:", STICKY_FIRST)
        st.write("Raw COLUMN_WIDTHS_PX_READONLY (type)", type(COLUMN_WIDTHS).__name__); st.code(COLUMN_WIDTHS)
        st.write("Column label overrides (if any)"); st.code(LABEL_OVERRIDES)


# -----------------------------
# Main
# -----------------------------
def main():
    # No title row; just render the table section
    render_view()
    # Ensure debug expander is at the very bottom of the page
    render_status_debug()

if __name__ == "__main__":
    main()
