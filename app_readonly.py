# app_readonly.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import io
import re
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
# Utilities
# =============================
def format_phone_display(value):
    """Return (xxx) xxx-xxxx for 10-digit inputs; otherwise original value."""
    if value is None:
        return value
    s = str(value)
    digits = re.sub(r"\D", "", s)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return s


def _get_secret(name: str, default: Optional[str | int | dict | bool] = None):
    """Prefer Streamlit secrets; tolerate env fallback only for simple strings."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    if isinstance(default, (str, type(None))):
        return os.getenv(name, default)
    return default


def _as_bool(v: Optional[str | bool], default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _to_int(v, default: int) -> int:
    """Coerce to int; tolerate '200px', ' 180 ', floats, etc."""
    try:
        if isinstance(v, bool):
            return default
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        s = str(v).strip()
        if not s:
            return default
        m = re.search(r"-?\d+", s)
        if not m:
            return default
        n = int(m.group(0))
        return n if n > 0 else default
    except Exception:
        return default


def _css_len_from_secret(v: str | int | None, fallback_px: int) -> str:
    """Accepts '0', '20', '18px', '1rem', '0.5em'."""
    if v is None:
        return f"{fallback_px}px"
    s = str(v).strip()
    if not s:
        return f"{fallback_px}px"
    if re.fullmatch(r"\d+(\.\d+)?(px|rem|em)?", s):
        return s if s.endswith(("px", "rem", "em")) else f"{s}px"
    return f"{fallback_px}px"


def _sanitize_url(url: str | None) -> str:
    if not url:
        return ""
    u = str(url).strip()
    if not re.match(r"^https?://", u, flags=re.I):
        u = "https://" + u
    return u


def _get_help_md() -> str:
    # 1) top-level
    try:
        top = (st.secrets.get("HELP_MD", "") or "").strip()
        if top:
            return top
    except Exception:
        pass
    # 2) nested in widths (common paste mistake)
    try:
        nested = st.secrets.get("COLUMN_WIDTHS_PX_READONLY", {})
        if isinstance(nested, dict):
            cand = (nested.get("HELP_MD", "") or "").strip()
            if cand:
                return cand
    except Exception:
        pass
    # 3) fallback
    return textwrap.dedent(
        """
        <style>
          .help-body p, .help-body li { font-size: 1rem; line-height: 1.45; }
          .help-body h1 { font-size: 1.25rem; margin: 0 0 6px; }
          .help-body h2 { font-size: 1.1rem;  margin: 10px 0 6px; }
          .help-body h3 { font-size: 1.0rem;  margin: 8px 0 4px; }
        </style>
        <div class="help-body">
          <h1>How to Use This List</h1>
          <p>Type in the search box, choose Sort, and download CSV/XLSX of your current view.</p>
        </div>
        """
    ).strip()


# =============================
# Page config
# =============================
PAGE_TITLE = _get_secret("page_title", "HCR Providers — Read-Only")
PAGE_MAX_WIDTH_PX = _to_int(_get_secret("page_max_width_px", 2300), 2300)
SIDEBAR_STATE = _get_secret("sidebar_state", "collapsed")
SHOW_DIAGS = _as_bool(_get_secret("READONLY_SHOW_DIAGS", False), False)
SHOW_STATUS = _as_bool(_get_secret("READONLY_SHOW_STATUS", False), False)

# secrets-driven padding (left + top)
PAD_LEFT_CSS = _css_len_from_secret(_get_secret("page_left_padding_px", "12"), 12)
TOP_PAD_PX = _to_int(_get_secret("page_top_padding_px", "10"), 10)
COMPACT = _as_bool(_get_secret("READONLY_COMPACT", True), True)

# Controls layout (we'll still respect the secret, but code below is single-row)
CONTROLS_LAYOUT = str(_get_secret("READONLY_CONTROLS_LAYOUT", "one_row") or "one_row").strip().lower()
if CONTROLS_LAYOUT not in ("two_row", "one_row"):
    CONTROLS_LAYOUT = "one_row"

# viewport rows (10–40 clamp)
def _viewport_rows() -> int:
    n = _to_int(_get_secret("READONLY_VIEWPORT_ROWS", 15), 15)
    return 10 if n < 10 else (40 if n > 40 else n)

VIEWPORT_ROWS = _viewport_rows()
ROW_PX = 32
HEADER_PX = 44
SCROLL_MAX_HEIGHT = HEADER_PX + ROW_PX * VIEWPORT_ROWS

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

# Ultra-compact CSS (header/toolbar hidden, tiny gaps, short controls)
st.markdown(
    f"""
    <style>
      /* Max width, zero top padding, left padding from secrets */
      [data-testid="stAppViewContainer"] .main .block-container {{
        max-width: {PAGE_MAX_WIDTH_PX}px;
        padding-top: {TOP_PAD_PX}px !important;
        padding-left: {PAD_LEFT_CSS};
      }}

      /* Hide Streamlit chrome to reclaim vertical space */
      [data-testid="stHeader"] {{ height:0 !important; min-height:0 !important; visibility:hidden !important; }}
      [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display:none !important; }}

      /* Make inter-block gaps tiny */
      [data-testid='stVerticalBlock'], [data-testid='stHorizontalBlock'] {{ gap: 2px !important; }}

      /* No label gap */
      [data-testid="stTextInput"] label,
      [data-testid="stSelectbox"] label {{ margin-bottom:0 !important; }}

      /* Shorter controls */
      [data-testid="stTextInput"] input {{ min-height:28px !important; padding:4px 8px !important; }}
      [data-baseweb="select"] {{ min-height:28px !important; }}
      [data-testid="stDownloadButton"] button,
      [data-testid='baseButton-secondary'] button,
      [data-testid='baseButton-primary'] button {{
        min-height:28px !important; padding:4px 8px !important; line-height:1.1 !important;
      }}

      /* Keep download labels single-line to save height */
      [data-testid="stDownloadButton"] button {{ white-space:nowrap !important; }}

      /* Table styling (slightly tighter) */
      .prov-table td, .prov-table th {{ padding:4px 6px; white-space:normal; word-break:break-word; vertical-align:top; border-bottom:1px solid #eee; }}
      .prov-table thead th {{ position:sticky; top:0; background:#fff; z-index:2; border-bottom:2px solid #ddd; }}
      .prov-wrap {{ overflow-x:auto; }}
      .prov-scroll {{ max-height: {SCROLL_MAX_HEIGHT}px; overflow-y:auto; }}

      /* Subtle search focus */
      [data-testid="stTextInput"] input {{ border:1px solid #d0d0d0 !important; }}
      [data-testid="stTextInput"]:focus-within input {{
        outline:none !important; border:1px solid #6aa0ff !important; box-shadow:0 0 0 2px rgba(106,160,255,.15) !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

if SHOW_DIAGS:
    st.caption(f"HELP_MD present: {'HELP_MD' in getattr(st, 'secrets', {})}")
    st.caption(
        "Turso secrets — URL: %s | TOKEN: %s"
        % (bool(_get_secret("TURSO_DATABASE_URL", "")), bool(_get_secret("TURSO_AUTH_TOKEN", "")))
    )
    st.caption(f"Viewport rows: {VIEWPORT_ROWS} (height ≈ {SCROLL_MAX_HEIGHT}px)")


# =============================
# Column labels & widths (SAFE)
# =============================
_readonly_labels_raw = _get_secret("READONLY_COLUMN_LABELS", {}) or {}
_col_widths_raw = _get_secret("COLUMN_WIDTHS_PX_READONLY", {}) or {}

READONLY_COLUMN_LABELS: Dict[str, str] = dict(_readonly_labels_raw)

_DEFAULT_WIDTHS = {
    "id": 40,
    "business_name": 180,
    "category": 175,
    "service": 120,
    "contact_name": 110,
    "phone": 120,
    "address": 220,
    "website": 160,
    "notes": 220,
    "keywords": 90,
}

_user_widths: Dict[str, int] = {}
for k, v in dict(_col_widths_raw).items():
    sk = str(k)
    if sk not in _DEFAULT_WIDTHS:
        continue
    _user_widths[sk] = _to_int(v, _DEFAULT_WIDTHS.get(sk, 140))

COLUMN_WIDTHS_PX_READONLY: Dict[str, int] = {**_DEFAULT_WIDTHS, **_user_widths}
for k, v in list(COLUMN_WIDTHS_PX_READONLY.items()):
    if not isinstance(v, int) or v <= 0:
        COLUMN_WIDTHS_PX_READONLY[k] = _DEFAULT_WIDTHS.get(k, 120)

STICKY_FIRST_COL: bool = False  # no pinned column
# Hide from display; still searchable (includes computed_keywords)
HIDE_IN_DISPLAY = {"id", "keywords", "computed_keywords", "created_at", "updated_at", "updated_by"}


# =============================
# Database engine
# =============================
def build_engine() -> Tuple[Engine, Dict[str, str]]:
    info: Dict[str, str] = {}

    raw_url = _get_secret("TURSO_DATABASE_URL", "") or ""
    token   = _get_secret("TURSO_AUTH_TOKEN", "") or ""
    embedded_path = os.path.abspath(_get_secret("EMBEDDED_DB_PATH", "vendors-embedded.db"))

    if raw_url and token:
        if raw_url.startswith("sqlite+libsql://"):
            host = raw_url.split("://", 1)[1].split("?", 1)[0]
            sync_url = f"libsql://{host}"
        elif raw_url.startswith("libsql://"):
            sync_url = raw_url.split("?", 1)[0]
        else:
            host = raw_url.split("://", 1)[-1].split("?", 1)[0]
            sync_url = f"libsql://{host}"

        sqlalchemy_url = f"sqlite+libsql:///{embedded_path}"

        engine = create_engine(
            sqlalchemy_url,
            connect_args={"auth_token": token, "sync_url": sync_url},
            pool_pre_ping=True,
            pool_recycle=300,
            pool_reset_on_return="commit",
        )
        with engine.connect() as c:
            c.exec_driver_sql("select 1;")

        info.update(
            {
                "using_remote": "true",
                "strategy": "embedded_replica",
                "sqlalchemy_url": sqlalchemy_url,
                "dialect": "sqlite",
                "driver": getattr(engine.dialect, "driver", ""),
                "sync_url": sync_url,
            }
        )
        return engine, info

    local_url = f"sqlite:///{os.path.abspath('./vendors.db')}"
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
    "id", "category", "service", "business_name", "contact_name", "phone",
    "address", "website", "notes", "keywords", "computed_keywords",
    "created_at", "updated_at", "updated_by",
]


def fetch_vendors(engine: Engine) -> pd.DataFrame:
    sql = "SELECT " + ", ".join(VENDOR_COLS) + " FROM vendors ORDER BY lower(business_name)"
    with engine.begin() as conn:
        df = pd.read_sql(sql_text(sql), conn)

    def _mk_anchor(href: str) -> str:
        if not href:
            return ""
        href = _sanitize_url(href.strip())
        return f'<a href="{html.escape(href, quote=True)}" target="_blank" rel="noopener noreferrer">Website</a>'

    if "website" in df.columns:
        df["website"] = df["website"].fillna("").astype(str).map(_mk_anchor)

    if "phone" in df.columns:
        df["phone"] = df["phone"].map(format_phone_display)

    for c in df.columns:
        if c not in ("id", "created_at", "updated_at"):
            df[c] = df[c].fillna("").astype(str)

    return df


# =============================
# Filtering
# =============================
def apply_global_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = (query or "").strip().lower()
    if not q:
        return df
    mask = pd.Series(False, index=df.index)
    for c in df.columns:
        s = df[c].astype(str).str.lower()
        mask |= s.str.contains(q, regex=False, na=False)
    return df[mask]


# =============================
# Rendering
# =============================
def _rename_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping = {k: READONLY_COLUMN_LABELS.get(k, k.replace("_", " ").title()) for k in df.columns}
    df2 = df.rename(columns={k: mapping[k] for k in df.columns})
    rev = {v: k for k, v in mapping.items()}
    return df2, rev


def _build_table_html(df: pd.DataFrame, sticky_first: bool) -> str:
    df_disp, rev = _rename_columns(df)
    headers = []
    for col in df_disp.columns:
        orig = rev[col]
        width = COLUMN_WIDTHS_PX_READONLY.get(orig, 140)
        headers.append(
            f'<th style="min-width:{width}px;max-width:{width}px;">{html.escape(col)}</th>'
        )
    thead = "<thead><tr>" + "".join(headers) + "</tr></thead>"

    rows_html: List[str] = []
    for _, row in df_disp.iterrows():
        tds: List[str] = []
        for col in df_disp.columns:
            orig = rev[col]
            val = row[col]
            width = COLUMN_WIDTHS_PX_READONLY.get(orig, 140)
            if orig == "website":
                cell_html = val
            else:
                cell_html = html.escape(str(val) if val is not None else "")
            tds.append(f'<td style="min-width:{width}px;max-width:{width}px;">{cell_html}</td>')
        rows_html.append("<tr>" + "".join(tds) + "</tr>")
    tbody = "<tbody>" + "".join(rows_html) + "</tbody>"
    return f'<div class="prov-wrap prov-scroll"><table class="prov-table">{thead}{tbody}</table></div>'


# =============================
# Main UI
# =============================
def main():
    engine, info = build_engine()
    if info.get("strategy") == "local_fallback":
        st.warning("Turso credentials not found. Running on local SQLite fallback (`./vendors.db`).")

    try:
        df_full = fetch_vendors(engine)
    except Exception as e:
        st.error(f"Failed to load vendors: {e}")
        return

    # ---------- Single-row controls: Search(40%) | CSV(15%) | XLSX(15%) | Sort(15%) | Order(15%) ----------
    c_search, c_csv, c_xlsx, c_sort, c_order = st.columns([8, 3, 3, 3, 3])
    
    # ==== BEGIN PATCH: search placeholder tweak (read-only app) ====
    with c_search:
        st.text_input(
            "Search",
            key="q",  # keep the same state key
            label_visibility="collapsed",
            placeholder="SEARCH     plumb → plumber, plumbing, plumbago",
            help="Case-insensitive substring match across all columns (including hidden computed_keywords).",
            autocomplete="off",
        )
    # ==== END PATCH ====

    # Filter first (search applies to full DF incl. computed_keywords)
    q = (st.session_state.get("q") or "")
    filtered_full = apply_global_search(df_full, q)

    # Columns we show (hide internal columns)
    disp_cols = [c for c in filtered_full.columns if c not in HIDE_IN_DISPLAY]
    df_disp_all = filtered_full[disp_cols]

    # --- safety default so df_disp_sorted always exists ---
    df_disp_sorted = df_disp_all.copy()

    # Sort controls
    def _label_for(col_key: str) -> str:
        return READONLY_COLUMN_LABELS.get(col_key, col_key.replace("_", " ").title())

    sortable_cols = [c for c in disp_cols if c != "website"]
    sort_labels = [_label_for(c) for c in sortable_cols]

    with c_sort:
        if sortable_cols:
            default_sort_col = "business_name" if "business_name" in sortable_cols else sortable_cols[0]
            default_label = _label_for(default_sort_col)
            chosen_label = st.selectbox(
                "Sort by",
                options=sort_labels,
                index=max(0, sort_labels.index(default_label)) if default_label in sort_labels else 0,
                key="sort_by_label",
                label_visibility="collapsed",
            )
        else:
            chosen_label = st.selectbox("Sort by", options=["(none)"], index=0, key="sort_by_label", label_visibility="collapsed")

    with c_order:
        order = st.selectbox(
            "Order",
            options=["Ascending", "Descending"],
            index=0,
            key="sort_order",
            label_visibility="collapsed",
        )

    # Compute sorted view now so downloads reflect the sorted view
    sort_col = None
    if sortable_cols and chosen_label in sort_labels:
        sort_col = sortable_cols[sort_labels.index(chosen_label)]
    ascending = (order == "Ascending")

    if sort_col is not None and sort_col in df_disp_all.columns and not df_disp_all.empty:
        keyfunc = (lambda s: s.str.lower()) if pd.api.types.is_string_dtype(df_disp_all[sort_col]) else None
        df_disp_sorted = df_disp_all.sort_values(
            by=sort_col,
            ascending=ascending,
            kind="mergesort",
            key=keyfunc
        )

    # ---- Download buttons (sorted view) — single-line labels to minimize height ----
    csv_df = df_disp_sorted.copy()
    if "website" in csv_df.columns and not csv_df.empty:
        csv_df["website"] = csv_df["website"].str.replace(r'.*href="([^"]+)".*', r"\1", regex=True)
    csv_buf = io.StringIO()
    if not csv_df.empty:
        csv_df.to_csv(csv_buf, index=False)
    with c_csv:
        st.download_button(
            "CSV",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name="providers.csv",
            mime="text/csv",
            type="secondary",
            use_container_width=True,
            disabled=csv_df.empty,
            help="Download current view as CSV",
        )

    excel_df = df_disp_sorted.copy()
    if "website" in excel_df.columns and not excel_df.empty:
        excel_df["website"] = excel_df["website"].str.replace(r'.*href="([^"]+)".*', r"\1", regex=True)
    xlsx_buf = io.BytesIO()
    if not excel_df.empty:
        with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
            excel_df.to_excel(writer, sheet_name="providers", index=False)
        xlsx_data = xlsx_buf.getvalue()
    else:
        xlsx_data = b""
    with c_xlsx:
        st.download_button(
            "XLSX",
            data=xlsx_data,
            file_name="providers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="secondary",
            use_container_width=True,
            disabled=excel_df.empty,
            help="Download current view as Excel",
        )

    # Optional status line (toggle via Secrets)
    if SHOW_STATUS:
        st.caption(f"{len(df_disp_sorted)} matching provider(s). Viewport rows: {VIEWPORT_ROWS}")
    # ---------------- Scrollable full table ----------------
    if df_disp_sorted.empty:
        st.info("No matching providers.")
    else:
        # small spacer to add room above headings
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(_build_table_html(df_disp_sorted, sticky_first=STICKY_FIRST_COL), unsafe_allow_html=True)

        # ==== BEGIN PATCH: small gap above Help/Tips (spacer only) ====
        # Add a small vertical spacer between the table and the Help/Tips box
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        # ==== END PATCH ====

    # Help/Tips expander — moved BELOW the table to free vertical space
    with st.expander("Help / Tips (click to expand)", expanded=False):
        st.markdown(_get_help_md(), unsafe_allow_html=True)

    # Debug
    st.divider()
    if st.button("Debug (status & secrets keys)", type="secondary"):
        dbg = {
            "DB (resolved)": info,
            "Secrets keys": sorted(list(getattr(st, 'secrets', {}).keys())) if hasattr(st, "secrets") else [],
            "Widths (effective)": COLUMN_WIDTHS_PX_READONLY,
            "Viewport rows": VIEWPORT_ROWS,
            "Scroll height px": SCROLL_MAX_HEIGHT,
            "controls_layout": CONTROLS_LAYOUT,
        }
        st.write(dbg)
        st.caption(f"HELP_MD present (top-level): {'HELP_MD' in getattr(st, 'secrets', {})}")
        try:
            nested = st.secrets.get("COLUMN_WIDTHS_PX_READONLY", {})
            has_nested_help = isinstance(nested, dict) and bool((nested.get("HELP_MD", "") or "").strip())
        except Exception:
            has_nested_help = False
        st.caption(f"HELP_MD present (nested in widths): {has_nested_help}")

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
