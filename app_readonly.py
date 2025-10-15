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

# ==== BEGIN: Boot-time page title from Secrets ====
def _safe_title(v, default="HCR Providers — Read-Only"):
    try:
        s = str(v).strip()
        return re.sub(r"[\x00-\x1f]+", "", s) or default
    except Exception:
        return default

try:
    _title_secret = st.secrets.get("page_title", "HCR Providers — Read-Only")
except Exception:
    _title_secret = "HCR Providers — Read-Only"

st.set_page_config(
    page_title=_safe_title(_title_secret),
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.caption("Read-only boot OK — reached post-page_config")

# ==== END: Boot-time page title from Secrets ====

# =============================
# Utilities
# =============================
def _get_secret(name: str, default: Optional[str | int | dict | bool] = None):
    """Prefer Streamlit secrets; tolerate env fallback only for simple strings."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    if isinstance(default, (str, type(None))):
        return os.getenv(name, default)  # env fallback for strings
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
    """
    Accepts e.g. "0", "20", "18px", "1rem", "0.5em".
    Returns a safe CSS length string; defaults to '{fallback_px}px' on bad input.
    """
    if v is None:
        return f"{fallback_px}px"
    s = str(v).strip()
    if not s:
        return f"{fallback_px}px"
    # numeric with optional unit
    if re.fullmatch(r"\d+(\.\d+)?(px|rem|em)?", s):
        if s.endswith(("px", "rem", "em")):
            return s
        return f"{s}px"
    return f"{fallback_px}px"

def _get_help_md() -> str:
    """Prefer top-level HELP_MD; if missing, also look under widths; else fallback."""
    # 1) top-level
    try:
        top = (st.secrets.get("HELP_MD", "") or "").strip()
        if top:
            return top
    except Exception:
        pass
    # 2) nested in [COLUMN_WIDTHS_PX_READONLY] (common paste mistake)
    try:
        nested = st.secrets.get("COLUMN_WIDTHS_PX_READONLY", {})
        if isinstance(nested, dict):
            cand = (nested.get("HELP_MD", "") or "").strip()
            if cand:
                return cand
    except Exception:
        pass
    # 3) built-in fallback
    return textwrap.dedent(
        """
        <style>
          .help-body p, .help-body li { font-size: 1rem; line-height: 1.45; }
          .help-body h1 { font-size: 1.4rem; margin: 0 0 6px; }
          .help-body h2 { font-size: 1.2rem; margin: 12px 0 6px; }
          .help-body h3 { font-size: 1.05rem; margin: 10px 0 6px; }
        </style>
        <div class="help-body">
          <h1>How to Use This List</h1>
          <p>Type in the search box, choose Sort, and download CSV/XLSX of your current view.</p>
        </div>
        """
    ).strip()

# =============================
# Secrets-driven page options
# =============================
PAGE_TITLE = _get_secret("page_title", "HCR Providers — Read-Only")
PAGE_MAX_WIDTH_PX = _to_int(_get_secret("page_max_width_px", 2300), 2300)
SIDEBAR_STATE = _get_secret("sidebar_state", "collapsed")
SHOW_DIAGS = _as_bool(_get_secret("READONLY_SHOW_DIAGS", False), False)
SHOW_STATUS = _as_bool(_get_secret("READONLY_SHOW_STATUS", False), False)

# Prioritized-search toggle (enabled by default; can turn off in secrets)
READONLY_PRIORITIZE_CKW = _as_bool(_get_secret("READONLY_PRIORITIZE_CKW", "true"), True)

# Cache TTL (seconds) for vendor list (default 120; override with READONLY_CACHE_TTL)
READONLY_CACHE_TTL = _to_int(_get_secret("READONLY_CACHE_TTL", 120), 120)

# Apply sidebar state retroactively (page title must stay as set_page_config)
try:
    if SIDEBAR_STATE not in ("collapsed", "expanded"):
        SIDEBAR_STATE = "collapsed"
except Exception:
    SIDEBAR_STATE = "collapsed"

# secrets-driven padding (matches admin app behavior)
PAD_LEFT_CSS = _css_len_from_secret(_get_secret("page_left_padding_px", "12"), 12)

# secrets-driven viewport rows (10–40 clamp)
def _viewport_rows() -> int:
    n = _to_int(_get_secret("READONLY_VIEWPORT_ROWS", 15), 15)
    if n < 10:
        return 10
    if n > 40:
        return 40
    return n

VIEWPORT_ROWS = _viewport_rows()
ROW_PX = 32   # approximate row height
HEADER_PX = 44
SCROLL_MAX_HEIGHT = HEADER_PX + ROW_PX * VIEWPORT_ROWS  # pixels

# ---- Help/Tips expander state (for Close button) ----
st.session_state.setdefault("help_open", False)

st.markdown(
    f"""
    <style>
      .block-container {{
        max-width: {PAGE_MAX_WIDTH_PX}px;
        padding-left: {PAD_LEFT_CSS};
      }}
      div[data-testid="stDataFrame"] table {{ white-space: nowrap; }}  /* harmless if not used */
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
      .prov-wrap {{ overflow-x: auto; }}
      /* vertical scroll viewport (mouse-wheel scroll to end) */
      .prov-scroll {{
        max-height: {SCROLL_MAX_HEIGHT}px;
        overflow-y: auto;
      }}
      .help-row {{ display: flex; align-items: center; gap: 12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Optional version banner (sidebar), toggle via READONLY_SHOW_STATUS
if SHOW_STATUS:
    try:
        import sys, requests, sqlalchemy  # noqa: F401
        _lib_ver = "n/a"
        try:
            import sqlalchemy_libsql as _lib  # noqa: F401
            _lib_ver = getattr(_lib, "__version__", "unknown")
        except Exception:
            pass
        st.sidebar.info(
            f"Versions | py {sys.version.split()[0]} | "
            f"streamlit {st.__version__} | "
            f"pandas {pd.__version__} | "
            f"SA {sqlalchemy.__version__} | "
            f"libsql {_lib_ver}"
        )
    except Exception as _e:
        st.sidebar.warning(f"Version banner failed: {_e}")

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
    # hidden columns still need sane widths if ever revealed
    "computed_keywords": 160,
    "created_at": 120,
    "updated_at": 120,
    "updated_by": 120,
    # internal-only (not displayed) but keep a default in case surfaced
    "website_url": 180,
}

# Only accept known width keys; ignore accidental extras (e.g., HELP_MD)
_user_widths: Dict[str, int] = {}
for k, v in dict(_col_widths_raw).items():
    sk = str(k)
    if sk not in _DEFAULT_WIDTHS:
        continue
    _user_widths[sk] = _to_int(v, _DEFAULT_WIDTHS.get(sk, 140))

COLUMN_WIDTHS_PX_READONLY: Dict[str, int] = {**_DEFAULT_WIDTHS, **_user_widths}

# Clamp invalids
for k, v in list(COLUMN_WIDTHS_PX_READONLY.items()):
    if not isinstance(v, int) or v <= 0:
        COLUMN_WIDTHS_PX_READONLY[k] = _DEFAULT_WIDTHS.get(k, 120)

# Do NOT pin first column (per request)
STICKY_FIRST_COL: bool = False

# Hide these in display; still searchable
HIDE_IN_DISPLAY = {
    "id",
    "keywords",
    "computed_keywords",
    "created_at",
    "updated_at",
    "updated_by",
    "website_url",  # internal plain URL used for search/export
}

# =============================
# Database engine
# =============================
def build_engine() -> Tuple[Engine, Dict[str, str]]:
    info: Dict[str, str] = {}

    raw_url = _get_secret("TURSO_DATABASE_URL", "") or ""
    token   = _get_secret("TURSO_AUTH_TOKEN", "") or ""
    embedded_path = os.path.abspath(_get_secret("EMBEDDED_DB_PATH", "vendors-embedded.db"))

    if raw_url and token:
        # sync_url must be libsql://... (strip sqlite+libsql:// and query if present)
        if raw_url.startswith("sqlite+libsql://"):
            host = raw_url.split("://", 1)[1].split("?", 1)[0]
            sync_url = f"libsql://{host}"
        elif raw_url.startswith("libsql://"):
            sync_url = raw_url.split("?", 1)[0]
        else:
            host = raw_url.split("://", 1)[-1].split("?", 1)[0]
            sync_url = f"libsql://{host}"

        # SQLAlchemy engine URL uses sqlite+libsql with a local embedded file
        sqlalchemy_url = f"sqlite+libsql:///{embedded_path}"

        engine = create_engine(
            sqlalchemy_url,
            connect_args={
                "auth_token": token,
                "sync_url": sync_url,
            },
            pool_pre_ping=True,
            pool_recycle=300,
            pool_reset_on_return="commit",
        )
        # probe (read-only connection; no explicit txn)
        with engine.connect() as c:
            c.exec_driver_sql("select 1;")

        def _mask(u: str) -> str:
            try:
                return "libsql://" + u.split("://", 1)[1].split("/", 1)[0]
            except Exception:
                return "libsql://"

        info.update(
            {
                "using_remote": "true",
                "strategy": "embedded_replica",
                "sqlalchemy_url": sqlalchemy_url,
                "dialect": "sqlite",
                "driver": getattr(engine.dialect, "driver", ""),
                "sync_url": _mask(sync_url),  # masked
                "embedded_path": embedded_path,
            }
        )
        return engine, info

    # Fallback (local dev only)
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
            "embedded_path": local_path,
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
    "computed_keywords",  # used for prioritized search; hidden from display
    "created_at",
    "updated_at",
    "updated_by",
]

@st.cache_data(ttl=READONLY_CACHE_TTL, show_spinner=False)
def fetch_vendors(_engine: Engine) -> pd.DataFrame:
    """
    Load vendors; cached briefly to reduce DB hits.
    Leading underscore on _engine tells Streamlit not to hash the Engine object.
    """
    sql = "SELECT " + ", ".join(VENDOR_COLS) + " FROM vendors ORDER BY lower(business_name)"
    with _engine.connect() as conn:  # read-only; no transaction
        df = pd.read_sql(sql_text(sql), conn)

    def _normalize_url(v: str) -> str:
        s = (v or "").strip()
        if not s:
            return ""
        if not (s.startswith("http://") or s.startswith("https://")):
            s = "https://" + s
        return s

    # Build a plain URL column for search/export; render anchor for UI
    if "website" in df.columns:
        df["website"] = df["website"].fillna("").astype(str)
        df["website_url"] = df["website"].map(_normalize_url)
        df["website"] = df["website_url"].map(
            lambda u: f'<a href="{html.escape(u, quote=True)}" target="_blank" rel="noopener noreferrer nofollow">Website</a>'
            if u else ""
        )

    # Coerce common text columns to string (but keep id/timestamps as-is)
    for c in df.columns:
        if c not in ("id", "created_at", "updated_at"):
            df[c] = df[c].fillna("").astype(str)
    return df

# =============================
# Filtering
# =============================
def apply_global_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Vectorized contains-any across all columns (case-insensitive)."""
    q = (query or "").strip().lower()
    if not q or df.empty:
        return df
    # Lowercase, string view of all columns
    hay = df.astype(str).apply(lambda s: s.str.lower(), axis=0)
    # contains per-column, reduce with any(axis=1)
    mask = hay.apply(lambda s: s.str.contains(q, regex=False, na=False), axis=0).any(axis=1)
    return df[mask]

def _filter_and_rank_by_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Prioritize rows that match in computed_keywords, then other matches.
    Case-insensitive substring match. Falls back gracefully if column is missing.
    """
    q = (query or "").strip().lower()
    if not q or df.empty:
        return df

    # Safe access / fallbacks
    ckw = df.get("computed_keywords", pd.Series([""] * len(df), index=df.index)).astype(str).str.lower()
    other = pd.concat(
        [
            df.get("business_name", ""),
            df.get("category", ""),
            df.get("service", ""),
            df.get("contact_name", ""),
            df.get("phone", ""),
            df.get("address", ""),
            df.get("website_url", ""),  # search plain URL, not HTML anchor
            df.get("notes", ""),
            df.get("keywords", ""),
        ],
        axis=1,
    ).astype(str).agg(" ".join, axis=1).str.lower()

    hit_ckw = ckw.str.contains(q, regex=False, na=False)
    hit_oth = other.str.contains(q, regex=False, na=False)
    any_hit = hit_ckw | hit_oth

    dfm = df.loc[any_hit].copy()
    dfm["_rank_ckw"] = (~hit_ckw.loc[dfm.index]).astype(int)  # 0 for ckw hit, 1 otherwise

    # Deterministic tie-break: business_name (case-insensitive), fallback empty string
    dfm["__bn"] = dfm.get("business_name", "").astype(str).str.lower()
    dfm.sort_values(
        by=["_rank_ckw", "__bn"],
        kind="mergesort",
        inplace=True,
    )
    dfm.drop(columns=["_rank_ckw", "__bn"], inplace=True, errors="ignore")
    return dfm

# =============================
# Rendering
# =============================
def _label_for(col_key: str) -> str:
    return READONLY_COLUMN_LABELS.get(col_key, col_key.replace("_", " ").title())

def _rename_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping = {k: _label_for(k) for k in df.columns}
    df2 = df.rename(columns={k: mapping[k] for k in df.columns})
    rev = {v: k for k, v in mapping.items()}
    return df2, rev

def _build_table_html(df: pd.DataFrame, sticky_first: bool) -> str:
    df_disp, rev = _rename_columns(df)
    headers = []
    for col in df_disp.columns:
        orig = rev[col]
        width = COLUMN_WIDTHS_PX_READONLY.get(orig, 140)
        headers.append(f'<th style="min-width:{width}px;max-width:{width}px;">{html.escape(col)}</th>')
    thead = "<thead><tr>" + "".join(headers) + "</tr></thead>"

    rows_html: List[str] = []
    for _, row in df_disp.iterrows():
        tds: List[str] = []
        for col in df_disp.columns:
            orig = rev[col]
            val = row[col]
            width = COLUMN_WIDTHS_PX_READONLY.get(orig, 140)
            if orig == "website":
                cell_html = val  # already an <a> anchor
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
# ==== BEGIN: Search row (input + Clear button on same line) ====
def _clear_search():
    st.session_state["q"] = ""
    st.rerun()

    # ==== BEGIN: Search row (input + Clear button on same line) ====
    def _clear_search():
        st.session_state["q"] = ""
        st.rerun()

    c_search, c_clearbtn = st.columns([12, 2])
    with c_search:
        st.text_input(
            "Search",  # non-empty for accessibility; label collapsed
            key="q",
            label_visibility="collapsed",
            placeholder="Search e.g., plumb, roofing, 'Inverness', phone digits, etc.",
            help="Case-insensitive; keyword hits appear first; matches across all columns.",
        )
    with c_clearbtn:
        st.button("Clear", type="secondary", use_container_width=True, on_click=_clear_search)
    # ==== END: Search row (input + Clear button on same line) ====

    q = st.session_state.get("q", "")

    if READONLY_PRIORITIZE_CKW:
        filtered_full = _filter_and_rank_by_query(df_full, q)
    else:
        filtered_full = apply_global_search(df_full, q)

    # Columns we show (hide internal columns)
    disp_cols = [c for c in filtered_full.columns if c not in HIDE_IN_DISPLAY]
    df_disp_all = filtered_full[disp_cols]

    # ----- Controls Row: Downloads/Sort -----
    # Exclude 'website' from sort choices (HTML anchors)
    sortable_cols = [c for c in disp_cols if c != "website"]
    sort_labels = [_label_for(c) for c in sortable_cols]

    # ==== BEGIN: Controls Row (no Help, no Clear — Clear moved next to Search) ====
    c_csv, c_xlsx, c_sort, c_order = st.columns([2, 2, 2, 2])
    # ==== END: Controls Row (no Help, no Clear — Clear moved next to Search) ====


    # Safe defaults when no sortable cols
    if len(sortable_cols) == 0:
        chosen_label = "Business Name"
        order = "Ascending"
        sort_col = None
        ascending = True
    else:
        default_sort_col = "business_name" if "business_name" in sortable_cols else sortable_cols[0]
        default_label = _label_for(default_sort_col)

        with c_sort:
            chosen_label = st.selectbox(
                "Sort by",
                options=sort_labels,
                index=max(0, sort_labels.index(default_label)) if default_label in sort_labels else 0,
                key="sort_by_label",
            )
        with c_order:
            order = st.selectbox(
                "Order",
                options=["Ascending", "Descending"],
                index=0,
                key="sort_order",
            )

        sort_col = sortable_cols[sort_labels.index(chosen_label)] if chosen_label in sort_labels else sortable_cols[0]
        ascending = (order == "Ascending")

    # Case-insensitive sort for text columns; stable sort keeps ties predictable
    if sort_col is not None and sort_col in df_disp_all.columns and not df_disp_all.empty:
        keyfunc = (lambda s: s.str.lower()) if pd.api.types.is_string_dtype(df_disp_all[sort_col]) else None
        df_disp_sorted = df_disp_all.sort_values(
            by=sort_col,
            ascending=ascending,
            kind="mergesort",
            key=keyfunc,
        )
    else:
        df_disp_sorted = df_disp_all.copy()

    # Downloads (use sorted view) — guard empty frames
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d-%H%M%S")

    # Prepare CSV with plain URLs (no HTML anchors)
    csv_df = df_disp_sorted.copy()
    if "website" in csv_df.columns and "website_url" in filtered_full.columns and not csv_df.empty:
        # Replace display anchor with plain URL for export
        csv_df["website"] = filtered_full.loc[csv_df.index, "website_url"].fillna("").astype(str)
    csv_buf = io.StringIO()
    if not csv_df.empty:
        csv_df.to_csv(csv_buf, index=False)
    with c_csv:
        st.download_button(
            "Download CSV",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name=f"providers_{ts}.csv",
            mime="text/csv",
            type="secondary",
            use_container_width=True,
            disabled=csv_df.empty,
        )
    # Prepare XLSX with plain URLs
    excel_df = df_disp_sorted.copy()
    if "website" in excel_df.columns and "website_url" in filtered_full.columns and not excel_df.empty:
        excel_df["website"] = filtered_full.loc[excel_df.index, "website_url"].fillna("").astype(str)

    xlsx_buf = io.BytesIO()
    if not excel_df.empty:
        with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
            excel_df.to_excel(writer, sheet_name="providers", index=False)
        xlsx_data = xlsx_buf.getvalue()
    else:
        xlsx_data = b""
    with c_xlsx:
        st.download_button(
            "Download XLSX",
            data=xlsx_data,
            file_name=f"providers_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="secondary",
            use_container_width=True,
            disabled=excel_df.empty,
        )

    # Always show tiny result count caption (keeps users oriented)
    st.caption(f"{len(df_disp_sorted)} result(s). Viewport rows: {VIEWPORT_ROWS}")

    # ==== BEGIN: Help / Tips (expander only) ====
    with st.expander("Help / Tips", expanded=False):
        st.markdown(_get_help_md(), unsafe_allow_html=True)
    # ==== END: Help / Tips (expander only) ====

    # ---------------- Scrollable full table (admin-style viewport) ----------------
    if df_disp_sorted.empty:
        st.info("No matching providers. Tip: try fewer words or click × to clear the search.")
    else:
        st.markdown(_build_table_html(df_disp_sorted, sticky_first=STICKY_FIRST_COL), unsafe_allow_html=True)

    # ==== BEGIN: Quick Stats (read-only) ====
    with st.expander("Quick Stats", expanded=False):
        try:
            nrows, ncols = df_disp_sorted.shape
            st.write({
                "rows_displayed": int(nrows),
                "columns": int(ncols),
                "unique_categories": int(df_disp_sorted["category"].nunique() if "category" in df_disp_sorted else 0),
                "unique_services": int(df_disp_sorted["service"].nunique() if "service" in df_disp_sorted else 0),
            })
        except Exception as _e:
            st.caption(f"Stats unavailable: {_e}")
    # ==== END: Quick Stats (read-only) ====

    # Debug / Diagnostics (shown only when explicitly enabled)
    if SHOW_DIAGS:
        st.divider()
        if st.button("Debug (status & secrets keys)", type="secondary"):
            dbg = {
                "DB (resolved)": info,
                "Secrets keys": sorted(list(getattr(st, "secrets", {}).keys())) if hasattr(st, "secrets") else [],
                "Widths (effective)": COLUMN_WIDTHS_PX_READONLY,
                "Viewport rows": VIEWPORT_ROWS,
                "Scroll height px": SCROLL_MAX_HEIGHT,
                "page_title_note": "Page title set at boot due to Streamlit order constraint.",
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
                with engine.connect() as conn:
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
    try:
        main()
    except Exception as e:
        import traceback, streamlit as st
        st.error("Top-level crash in app_readonly.py")
        st.code("".join(traceback.format_exc()))
