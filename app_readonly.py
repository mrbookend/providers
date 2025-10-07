# app_readonly.py — Providers Read‑Only (tuned to match Admin conventions)
# Traits:
# - Wide layout via secrets (page_title, page_max_width_px, sidebar_state)
# - Turso/libSQL first; optional guarded SQLite fallback for local dev
# - Global search (substring) across fields; NO per‑column filters
# - HTML table with true pixel widths from secrets, cell wrapping, optional sticky first column
# - Clickable website links (scheme auto‑added)
# - CSV download: filtered + full
# - Compact Debug button at bottom with engine + schema snapshot

from __future__ import annotations

import os
import html
from typing import Dict, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine
import sqlalchemy_libsql  # registers 'sqlite+libsql' dialect entrypoint
import streamlit.components.v1 as components

# -----------------------------
# Page layout FIRST
# -----------------------------

def _read_secret_early(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)

PAGE_TITLE = _read_secret_early("page_title", "Providers")
PAGE_MAX_WIDTH_PX = int(_read_secret_early("page_max_width_px", 2000))
SIDEBAR_STATE = _read_secret_early("sidebar_state", "expanded")

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

st.markdown(
    f"""
    <style>
      .block-container {{ max-width: {PAGE_MAX_WIDTH_PX}px; }}
      table.providers-grid {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
      table.providers-grid th, table.providers-grid td {{
        border: 1px solid #ddd; padding: 6px; vertical-align: top; word-wrap: break-word; white-space: normal;
      }}
      table.providers-grid th {{ position: sticky; top: 0; background: #f7f7f7; z-index: 2; }}
      .wrap {{ white-space: normal; word-wrap: break-word; overflow-wrap: anywhere; }}
      .sticky-first {{ position: sticky; left: 0; background: #fff; z-index: 1; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Secrets helpers
# -----------------------------

def _get_secret(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)

# Column widths (px) for the read-only grid
RAW_COL_WIDTHS = _get_secret("COLUMN_WIDTHS_PX_READONLY") or _get_secret("COLUMN_WIDTHS_PX") or {
    "id": 40,
    "business_name": 160,
    "category": 170,
    "service": 110,
    "contact_name": 110,
    "phone": 110,
    "address": 200,
    "website": 180,
    "notes": 220,
    "keywords": 120,
}

HELP_TITLE = _get_secret("READONLY_HELP_TITLE", "Provider Help / Tips")
HELP_MD = _get_secret("READONLY_HELP_MD", "")
STICKY_FIRST = str(_get_secret("READONLY_STICKY_FIRST_COL", "0")).lower() in ("1", "true", "yes")
HIDE_ID = str(_get_secret("READONLY_HIDE_ID", "1")).lower() in ("1", "true", "yes")
ALLOW_SQLITE_FALLBACK = str(_get_secret("ALLOW_SQLITE_FALLBACK", "0")).lower() in ("1", "true", "yes")

# -----------------------------
# Engine builder (Turso/libSQL first)
# -----------------------------

def build_engine() -> Tuple[Engine, Dict[str, str]]:
    turso_url = _get_secret("TURSO_DATABASE_URL")
    turso_token = _get_secret("TURSO_AUTH_TOKEN")
    info: Dict[str, str] = {
        "using_remote": False,
        "dialect": "",
        "driver": "",
        "sqlalchemy_url": "",
        "sync_url": turso_url or "",
    }

    def _fallback_sqlite(reason: str) -> Tuple[Engine, Dict[str, str]]:
        if not ALLOW_SQLITE_FALLBACK:
            st.error(
                reason
                + " Also, SQLite fallback is disabled. Ensure `sqlalchemy-libsql==0.2.0` is installed and DSN uses `sqlite+libsql://`. "
                  "Or set `ALLOW_SQLITE_FALLBACK=true` for local/dev only."
            )
            st.stop()
        local_path = os.environ.get("LOCAL_SQLITE_PATH", "vendors.db")
        e = create_engine(f"sqlite:///{local_path}")
        info.update({"using_remote": False, "sqlalchemy_url": f"sqlite:///{local_path}"})
        try:
            info["dialect"] = e.dialect.name
            info["driver"] = getattr(e.dialect, "driver", "")
        except Exception:
            pass
        return e, info

    if turso_url and turso_token:
        # Normalize DSN safely without collapsing sqlite file paths.
        dsn = str(turso_url).strip()

        # Accept both forms: libsql://…  → sqlite+libsql://…
        if dsn.startswith("libsql://"):
            dsn = "sqlite+libsql://" + dsn.split("://", 1)[1]

        # If it's a local file DSN and someone used a single slash, fix it to triple slashes:
        # sqlite+libsql:/vendors-embedded.db → sqlite+libsql:///vendors-embedded.db
        if dsn.startswith("sqlite+libsql:/") and not dsn.startswith("sqlite+libsql://"):
            dsn = "sqlite+libsql:///" + dsn.split(":/", 1)[1].lstrip("/")

        # Only enforce 'secure=true' for REMOTE DSNs (host present). For embedded file DSNs, DO NOT add it.
        is_file_dsn = dsn.startswith("sqlite+libsql:///")
        if not is_file_dsn:
            if "secure=" not in dsn.lower():
                dsn += ("&secure=true" if "?" in dsn else "?secure=true")
        else:
            # If someone mistakenly added secure=true to a file DSN, strip it to avoid DBAPI kwarg errors
            if "secure=" in dsn.lower():
                dsn = dsn.replace("&secure=true", "").replace("?secure=true", "?").replace("?&", "?")
                if dsn.endswith("?"):
                    dsn = dsn[:-1]


        try:
            e = create_engine(
                dsn,
                connect_args={"auth_token": str(turso_token)},
                pool_pre_ping=True,
                pool_recycle=180,
            )
            # Probe to force plugin load and catch auth/network errors
            with e.connect() as conn:
                conn.execute(sql_text("SELECT 1"))
            info.update({"using_remote": True, "sqlalchemy_url": dsn})
            try:
                info["dialect"] = e.dialect.name
                info["driver"] = getattr(e.dialect, "driver", "")
            except Exception:
                pass
            return e, info
        except Exception as ex:
            name = type(ex).__name__
            msg = f"Turso init failed ({name}: {ex})"
            return _fallback_sqlite(
                "SQLAlchemy couldn't load the 'libsql' dialect or connect. "
                "Verify `sqlalchemy-libsql==0.2.0` is installed and DSN uses `sqlite+libsql://`. "
                f"Details: {msg}"
            )

    # No Turso credentials at all
    return _fallback_sqlite("TURSO_DATABASE_URL / TURSO_AUTH_TOKEN missing.")


engine, engine_info = build_engine()

if not engine_info.get("using_remote"):
    st.warning("Running on local SQLite fallback (remote DB unavailable or disabled).", icon="⚠️")

# -----------------------------
# Data helpers
# -----------------------------

@st.cache_data(ttl=60)
def fetch_df(sql: str, params: Dict | None = None) -> pd.DataFrame:
    # Use SQLAlchemy execute instead of pandas.read_sql to avoid libsql ValueError paths
    with engine.connect() as conn:
        result = conn.execute(sql_text(sql), params or {})
        rows = result.fetchall()
        cols = result.keys()
    return pd.DataFrame(rows, columns=list(cols))

def vendors_df() -> pd.DataFrame:
    sql = (
        "SELECT id, business_name, category, service, contact_name, phone, address, website, notes, keywords "
        "FROM vendors ORDER BY lower(business_name)"
    )
    return fetch_df(sql)

# -----------------------------
# Table renderer (HTML)
# -----------------------------

def render_html_table(df: pd.DataFrame, sticky_first_col: bool = False) -> None:
    if df.empty:
        st.info("No rows match your filter.")
        return

    cols = list(df.columns)
    widths = [RAW_COL_WIDTHS.get(c, 120) for c in cols]

    ths = []
    for i, c in enumerate(cols):
        style = f"min-width:{widths[i]}px;max-width:{widths[i]}px;"
        extra = " sticky-first" if sticky_first_col and i == 0 else ""
        ths.append(f'<th class="wrap{extra}" style="{style}">{html.escape(str(c))}</th>')

    trs = []
    for _, row in df.iterrows():
        tds = []
        for i, c in enumerate(cols):
            val = row[c]
            txt = "" if pd.isna(val) else str(val)
            if c == "website" and txt:
                safe = html.escape(txt)
                href = safe if safe.startswith("http") else f"https://{safe}"
                inner = f'<a href="{href}" target="_blank" rel="noopener">{safe}</a>'
            else:
                inner = html.escape(txt)
            style = f"min-width:{widths[i]}px;max-width:{widths[i]}px;"
            extra = " sticky-first" if sticky_first_col and i == 0 else ""
            tds.append(f'<td class="wrap{extra}" style="{style}">{inner}</td>')
        trs.append("<tr>" + "".join(tds) + "</tr>")

    html_table = f"<table class='providers-grid'><thead><tr>{''.join(ths)}</tr></thead><tbody>{''.join(trs)}</tbody></table>"
    components.html(html_table, height=500, scrolling=True)


# -----------------------------
# Main UI
# -----------------------------

def main():
    if HELP_MD:
        with st.expander(HELP_TITLE or "Provider Help / Tips", expanded=False):
            st.markdown(HELP_MD)

    # Global search input directly below help
    q = st.text_input("Search (partial match across most fields)", placeholder="e.g., plumb or 210-555-…")

    df = vendors_df()
    if HIDE_ID and "id" in df.columns:
        df = df.drop(columns=["id"])  # hide id in read-only view by default

    if q:
        ql = q.lower().strip()
        # Filter across visible columns
        mask = pd.Series(False, index=df.index)
        for col in [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("string")]:
            mask = mask | df[col].astype(str).str.lower().str.contains(ql, na=False)
        df = df[mask].copy()

    render_html_table(df, sticky_first_col=STICKY_FIRST)

    # Downloads
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download filtered as providers.csv", df.to_csv(index=False).encode("utf-8"), "providers.csv", "text/csv")
    with c2:
        full = vendors_df()
        if HIDE_ID and "id" in full.columns:
            full = full.drop(columns=["id"])  # symmetry with view
        st.download_button("Download FULL export (all rows)", full.to_csv(index=False).encode("utf-8"), "providers_full.csv", "text/csv")

    # Debug button at bottom
    if st.button("Show Debug / Status", key="dbg_btn_ro"):
        st.write("Status & Secrets (debug)")
        st.json({
            "using_remote": engine_info.get("using_remote"),
            "strategy": "embedded_replica" if engine_info.get("driver") == "libsql" else "sqlite",
            "sqlalchemy_url": engine_info.get("sqlalchemy_url"),
            "dialect": engine_info.get("dialect"),
            "driver": engine_info.get("driver"),
            "sync_url": engine_info.get("sync_url"),
        })
        try:
            probe = {
                "vendors_columns": fetch_df("PRAGMA table_info(vendors)")["name"].tolist(),
                "categories_columns": fetch_df("PRAGMA table_info(categories)")["name"].tolist(),
                "services_columns": fetch_df("PRAGMA table_info(services)")["name"].tolist(),
            }
            cnts = fetch_df(
                "SELECT (SELECT COUNT(1) FROM vendors) AS vendors, (SELECT COUNT(1) FROM categories) AS categories, (SELECT COUNT(1) FROM services) AS services"
            ).iloc[0].to_dict()
            probe["counts"] = cnts
            st.json(probe)
        except Exception as e:
            st.error(f"Probe failed: {e}")


if __name__ == "__main__":
    main()
