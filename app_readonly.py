# app_readonly.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import re
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# ---- register libsql dialect (must be AFTER "import streamlit as st") ----
try:
    import sqlalchemy_libsql  # noqa: F401  (registers 'sqlite+libsql' dialect)
except Exception:
    pass

# =============================
# Secrets & Page Config
# =============================
def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return default

PAGE_TITLE = _get_secret("page_title", "HCR Providers — Read-Only")
PAGE_MAX_WIDTH_PX = int(_get_secret("page_max_width_px", "2300"))
SIDEBAR_STATE = _get_secret("sidebar_state", "collapsed")

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

st.markdown(
    f"""
    <style>
      .block-container {{
        max-width: {PAGE_MAX_WIDTH_PX}px !important;
      }}
      table.providers td, table.providers th {{
        vertical-align: top;
        padding: 6px 8px;
        border-bottom: 1px solid #eee;
        word-wrap: break-word;
        white-space: normal;
      }}
      table.providers th {{
        position: sticky;
        top: 0;
        background: #fafafa;
        z-index: 1;
      }}
      /* Stick first column */
      table.providers td:first-child, table.providers th:first-child {{
        position: sticky;
        left: 0;
        background: #fff;
        z-index: 2;
      }}
      /* Limit column widths (px) for wrap/auto-row-height effect */
      td.col-category     {{ max-width: 175px; }}
      td.col-service      {{ max-width: 140px; }}
      td.col-business     {{ max-width: 220px; }}
      td.col-contact      {{ max-width: 140px; }}
      td.col-address      {{ max-width: 280px; }}
      td.col-website      {{ max-width: 220px; }}
      td.col-notes        {{ max-width: 320px; }}
      td.col-keywords     {{ max-width: 160px; }}
      /* Make links look like links but avoid nested-anchor issues */
      a.safe-link {{ text-decoration: underline; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Engine & Data helpers
# =============================
def get_engine() -> tuple[Engine, Dict[str, str]]:
    """Create SQLAlchemy engine from secrets; fallback to local SQLite vendors.db read-only."""
    status = {}
    url = _get_secret("TURSO_DATABASE_URL")
    token = _get_secret("TURSO_AUTH_TOKEN")
    engine: Engine

    if url and token:
        status["backend"] = "libsql"
        status["dsn"] = "sqlite+libsql://<redacted-host>"
        try:
            engine = create_engine(url, connect_args={"auth_token": token})
            with engine.connect() as conn:
                conn.execute(sql_text("SELECT 1"))
            status["connected"] = "true"
            return engine, status
        except Exception as e:
            status["error"] = f"libsql connect failed: {type(e).__name__}"
            # fall through to local

    # Local fallback (read-only semantics depend on how app uses it; we only SELECT)
    status.setdefault("backend", "sqlite")
    status["dsn"] = "sqlite:///vendors-embedded.db"
    engine = create_engine("sqlite:///vendors-embedded.db")
    try:
        with engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))
        status["connected"] = "true"
    except Exception as e:
        status["error"] = f"sqlite connect failed: {type(e).__name__}"
    return engine, status

@st.cache_data(ttl=30)
def load_vendors_df() -> pd.DataFrame:
    engine, _ = get_engine()
    query = """
        SELECT id, category, service, business_name, contact_name, phone, address, website, notes, keywords,
               created_at, updated_at, updated_by
        FROM vendors
        ORDER BY lower(trim(category)) ASC, lower(trim(service)) ASC, lower(trim(business_name)) ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql(sql_text(query), conn)
    # Types & display prep
    for col in ("category","service","business_name","contact_name","address","website","notes","keywords","updated_by"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    if "phone" in df.columns:
        df["phone"] = df["phone"].fillna("").astype(str).map(_fmt_phone)
    return df

def _fmt_phone(p: str) -> str:
    digits = re.sub(r"\D", "", p or "")
    return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}" if len(digits) == 10 else p

def _safe_link(url: str) -> str:
    """Render a safe anchor; auto-prepend scheme; avoid nested anchor issues in Streamlit."""
    if not url:
        return ""
    u = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
        u = "https://" + u
    # simple sanitize
    u = u.replace('"', "%22")
    return f'<a class="safe-link" href="{u}" target="_blank" rel="noopener noreferrer">{u}</a>'

# =============================
# UI
# =============================
st.title("HCR Providers (Read-Only)")

# Help / Tips (from secrets or built-in)
HELP_MD = _get_secret("HELP_MD", """
### How to use this list
- Use the **Search** box to match any word or partial word across Provider, Category, Service, Contact, Address, Website, Notes, or Keywords.
- Click any column header to sort ascending/descending.
- Use **Download** at the bottom to export CSV or Excel.
""")

with st.expander("Provider Help / Tips", expanded=False):
    st.markdown(HELP_MD)

# Global search
search = st.text_input("Search (matches across most columns)", "", placeholder="e.g., plumber, roofing, 78255, 'Alexander'")
df = load_vendors_df()

def _filter_df(_df: pd.DataFrame, q: str) -> pd.DataFrame:
    if not q:
        return _df
    needles = [w.strip() for w in q.split() if w.strip()]
    if not needles:
        return _df
    cols = [c for c in ["business_name","category","service","contact_name","address","website","notes","keywords","phone"] if c in _df.columns]
    hay = _df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    mask = pd.Series(True, index=_df.index)
    for n in needles:
        mask &= hay.str.contains(re.escape(n.lower()), na=False)
    return _df[mask]

fdf = _filter_df(df, search)

# Sort controls
sort_cols = [c for c in ["category","service","business_name","contact_name","address","website","notes","keywords","phone","updated_at"] if c in fdf.columns]
default_sort = "category" if "category" in sort_cols else (sort_cols[0] if sort_cols else None)
left, right = st.columns([3,1])
with left:
    sort_by = st.selectbox("Sort by", sort_cols, index=(sort_cols.index(default_sort) if default_sort in sort_cols else 0))
with right:
    ascending = st.toggle("Ascending", value=True)

if sort_by:
    fdf = fdf.sort_values(by=[sort_by], ascending=ascending, kind="stable")

# Build HTML table (wrap & sticky headers; sticky first col = ID)
def _render_table(_df: pd.DataFrame) -> str:
    def td(cls: str, val: str) -> str:
        return f'<td class="{cls}">{val}</td>'

    headers = []
    order = [c for c in ["id","category","service","business_name","contact_name","phone","address","website","notes","keywords"] if c in _df.columns]
    labels = {
        "id": "ID",
        "category": "Category",
        "service": "Service",
        "business_name": "Provider",
        "contact_name": "Contact",
        "phone": "Phone",
        "address": "Address",
        "website": "Website",
        "notes": "Notes",
        "keywords": "Keywords",
    }
    headers.append("<tr>" + "".join(f"<th>{labels.get(c,c)}</th>" for c in order) + "</tr>")

    rows = []
    for _, r in _df.iterrows():
        cells = []
        for c in order:
            v = r.get(c, "")
            if c == "website":
                v = _safe_link(str(v))
            else:
                v = ("" if pd.isna(v) else str(v))
            cls = {
                "id":"col-id",
                "category":"col-category",
                "service":"col-service",
                "business_name":"col-business",
                "contact_name":"col-contact",
                "phone":"col-phone",
                "address":"col-address",
                "website":"col-website",
                "notes":"col-notes",
                "keywords":"col-keywords",
            }.get(c, "")
            cells.append(td(cls, v))
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return f'<table class="providers"><thead>{"".join(headers)}</thead><tbody>{"".join(rows)}</tbody></table>'

st.markdown(_render_table(fdf), unsafe_allow_html=True)

# Downloads
def _to_excel_bytes(_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        _df.to_excel(xw, index=False, sheet_name="providers")
    buf.seek(0)
    return buf.read()

st.divider()
dl1, dl2, dl3 = st.columns([1,1,6])
with dl1:
    st.download_button("Download CSV", fdf.to_csv(index=False).encode("utf-8"), file_name="providers.csv", mime="text/csv")
with dl2:
    st.download_button("Download Excel", _to_excel_bytes(fdf), file_name="providers.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Debug
st.divider()
if st.button("Show Debug (engine & schema)"):
    engine, status = get_engine()
    st.write({"Status & Secrets (debug)": {
        "backend": status.get("backend"),
        "dsn": status.get("dsn"),
        "connected": status.get("connected", "false"),
    }})
    try:
        with engine.connect() as conn:
            probe = {}
            for tbl in ("vendors","categories","services"):
                try:
                    cols = pd.read_sql(sql_text(f"PRAGMA table_info({tbl})"), conn)["name"].tolist()
                    cnt  = pd.read_sql(sql_text(f"SELECT COUNT(*) AS c FROM {tbl}"), conn)["c"].iat[0]
                    probe[f"{tbl}_columns"] = cols
                    probe[f"{tbl}_count"] = cnt
                except Exception:
                    pass
        st.write(probe)
    except Exception as e:
        st.write({"debug_error": str(e)})
