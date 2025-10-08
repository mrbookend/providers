# -*- coding: utf-8 -*-
# app_readonly.py - Providers Read-Only (embedded replica first, remote failover)
from __future__ import annotations

import os
import html
import time
from typing import Dict, Tuple, Any, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import sqlalchemy_libsql  # registers 'sqlite+libsql' dialect entrypoint


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

PAGE_TITLE = _read_secret_early("page_title", "Vendors Directory")
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
HELP_MD = _get_secret("READONLY_HELP_MD", "Global search matches any part of a word across most columns.")
STICKY_FIRST = str(_get_secret("READONLY_STICKY_FIRST_COL", "0")).lower() in ("1", "true", "yes")
HIDE_ID = str(_get_secret("READONLY_HIDE_ID", "1")).lower() in ("1", "true", "yes")
ALLOW_SQLITE_FALLBACK = str(_get_secret("ALLOW_SQLITE_FALLBACK", "0")).lower() in ("1", "true", "yes")


# -----------------------------
# Engine builder (Turso/libSQL first)
# -----------------------------
def _normalize_dsn(raw: str) -> tuple[str, bool]:
    """Normalize DSN. Returns (dsn, is_embedded)."""
    dsn = (raw or "").strip()

    # Strip accidental outer quotes
    if (dsn.startswith("'") and dsn.endswith("'")) or (dsn.startswith('"') and dsn.endswith('"')):
        dsn = dsn[1:-1].strip()

    # libsql://host -> sqlite+libsql://host
    if dsn.startswith("libsql://"):
        dsn = "sqlite+libsql://" + dsn.split("://", 1)[1]

    # sqlite+libsql:/file.db -> sqlite+libsql:///file.db
    if dsn.startswith("sqlite+libsql:/") and not dsn.startswith("sqlite+libsql://"):
        dsn = "sqlite+libsql:///" + dsn.split(":/", 1)[1].lstrip("/")

    is_embedded = dsn.startswith("sqlite+libsql:///")

    if is_embedded:
        # Ensure sync_url uses HTTPS (avoid 308s)
        if "sync_url=libsql://" in dsn:
            dsn = dsn.replace("sync_url=libsql://", "sync_url=https://")

        # Split file part and query
        after = dsn.split("sqlite+libsql:///", 1)[1]
        if "?" in after:
            path_part, q = after.split("?", 1)
            q = "?" + q
        else:
            path_part, q = after, ""

        # Absolute path? keep it. Otherwise map to /mount/data/<file>
        if path_part.startswith("/"):
            abs_path = path_part  # e.g. /mount/data/vendors-embedded.db
        else:
            abs_path = "/mount/data/" + path_part.lstrip("/")

        # Rebuild DSN with absolute path (no double-prefixing)
        dsn = "sqlite+libsql:///" + abs_path.lstrip("/") + q

        # Remove meaningless secure=true on embedded DSNs
        if "secure=" in dsn.lower():
            dsn = dsn.replace("&secure=true", "").replace("?secure=true", "?").replace("?&", "?")
            if dsn.endswith("?"):
                dsn = dsn[:-1]
    else:
        # Host DSN: ensure secure=true
        if "secure=" not in dsn.lower():
            dsn += ("&secure=true" if "?" in dsn else "?secure=true")

    return dsn, is_embedded


def _fallback_sqlite(reason: str) -> tuple[Engine, Dict[str, str]]:
    if not ALLOW_SQLITE_FALLBACK:
        st.error(
            reason
            + " Also, SQLite fallback is disabled. Ensure `sqlalchemy-libsql==0.2.0` is installed and DSN uses"
              " `sqlite+libsql://`. Or set `ALLOW_SQLITE_FALLBACK=true` for local/dev only."
        )
        st.stop()
    local_path = os.environ.get("LOCAL_SQLITE_PATH", "vendors.db")
    e = create_engine(f"sqlite:///{local_path}")
    info: Dict[str, str] = {
        "using_remote": False,
        "dialect": e.dialect.name,
        "driver": getattr(e.dialect, "driver", ""),
        "sqlalchemy_url": f"sqlite:///{local_path}",
        "sync_url": "",
        "strategy": "local_sqlite",
    }
    return e, info


def _build_engine_inner() -> tuple[Engine, Dict[str, str]]:
    url = _get_secret("TURSO_DATABASE_URL")
    token = _get_secret("TURSO_AUTH_TOKEN")

    if not url or not token:
        return _fallback_sqlite("TURSO_DATABASE_URL / TURSO_AUTH_TOKEN missing.")

    dsn, is_embedded = _normalize_dsn(str(url))
    try:
        e = create_engine(
            dsn,
            connect_args={"auth_token": str(token)},
            pool_pre_ping=True,
            pool_recycle=180,
        )
        with e.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        info: Dict[str, str] = {
            "using_remote": True,
            "dialect": e.dialect.name,
            "driver": getattr(e.dialect, "driver", ""),
            "sqlalchemy_url": dsn,
            "sync_url": dsn,  # informational
            "strategy": "embedded_replica" if is_embedded else "direct",
        }
        return e, info
    except Exception as ex:
        name = type(ex).__name__
        msg = f"Turso init failed ({name}: {ex})"
        return _fallback_sqlite(
            "SQLAlchemy couldn't load the 'libsql' dialect or connect. "
            "Verify `sqlalchemy-libsql==0.2.0` is installed and DSN uses `sqlite+libsql://`. "
            f"Details: {msg}"
        )


@st.cache_resource
def get_engine_and_info() -> tuple[Engine, Dict[str, str]]:
    return _build_engine_inner()


# Build engine now
engine, engine_info = get_engine_and_info()


# -----------------------------
# Embedded replica warm-up + automatic failover to direct remote
# -----------------------------
def _extract_sync_host_from_dsn(dsn: str) -> str | None:
    """Looks for sync_url=... in the DSN and returns host (no scheme, no path)."""
    try:
        qpos = dsn.find("?")
        if qpos == -1:
            return None
        for part in dsn[qpos + 1:].split("&"):
            if not part:
                continue
            k, _, v = part.partition("=")
            if k != "sync_url":
                continue
            v = v.strip()
            if "://" in v:
                v = v.split("://", 1)[1]
            v = v.split("/", 1)[0]
            return v or None
    except Exception:
        return None


def _warmup_embedded_replica(max_wait_seconds: float = 30.0) -> bool:
    """Touch schema and wait for vendors to appear; True if present, else False."""
    url = str(engine_info.get("sqlalchemy_url", ""))
    if not url.startswith("sqlite+libsql:///"):
        return True  # warm-up only applies to embedded file DSNs
    try:
        with engine.connect() as conn:
            # Touch the schema to trigger replication
            conn.exec_driver_sql("SELECT name FROM sqlite_master LIMIT 1")
            deadline = time.time() + max_wait_seconds
            while time.time() < deadline:
                row = conn.exec_driver_sql(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    ("vendors",),
                ).fetchone()
                if row:
                    return True
                time.sleep(0.5)
    except Exception as e:
        st.warning(f"Replica warm-up warning: {e}")
    return False


def _failover_to_direct_remote_if_needed() -> None:
    global engine  # we rebind engine here if failover is needed

    # If embedded vendors table is still missing, switch engine to direct remote.
    url = str(engine_info.get("sqlalchemy_url", ""))
    if not url.startswith("sqlite+libsql:///"):
        return  # not embedded; nothing to do

    # Check table presence one more time; if present, do nothing
    try:
        with engine.connect() as conn:
            row = conn.exec_driver_sql(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                ("vendors",),
            ).fetchone()
            if row:
                return
    except Exception:
        pass  # ignore and proceed to failover attempt

    sync_host = _extract_sync_host_from_dsn(url)
    if not sync_host:
        st.error("Embedded replica not ready and no sync_url host found; cannot fail over.")
        return

    remote_dsn = f"sqlite+libsql://{sync_host}?secure=true"
    try:
        remote_token = str(_get_secret("TURSO_AUTH_TOKEN") or "")
        if not remote_token:
            st.error("Embedded replica not ready and TURSO_AUTH_TOKEN missing; cannot fail over.")
            return

        # Build and probe a remote engine
        e = create_engine(
            remote_dsn,
            connect_args={"auth_token": remote_token},
            pool_pre_ping=True,
            pool_recycle=180,
        )
        with e.connect() as conn:
            conn.exec_driver_sql("SELECT 1")

        # Swap global engine and update info so downstream code uses remote
        engine = e
        engine_info.update(
            {
                "using_remote": True,
                "sqlalchemy_url": remote_dsn,
                "dialect": "sqlite",
                "driver": "libsql",
                "sync_url": remote_dsn,
                "strategy": "failover_remote",
            }
        )
        if not st.session_state.get("ro_failover_warned"):
            st.session_state["ro_failover_warned"] = True
            st.warning(
                "Embedded replica not ready; temporarily serving from direct remote. "
                "Reads are live; the app will use the embedded replica again once it catches up."
            )
        try:
            st.cache_data.clear()
        except Exception:
            pass

    except Exception as ex:
        st.error(f"Failover to remote DB failed: {type(ex).__name__}: {ex}")


def _maybe_return_to_embedded() -> None:
    """If we are in failover_remote and the embedded replica is ready, switch back."""
    global engine

    if engine_info.get("strategy") != "failover_remote":
        return

    embedded_dsn = str(_get_secret("TURSO_DATABASE_URL") or "").strip()
    if not embedded_dsn:
        return

    # Normalize to embedded DSN form
    if embedded_dsn.startswith("libsql://"):
        embedded_dsn = "sqlite+libsql://" + embedded_dsn.split("://", 1)[1]
    if embedded_dsn.startswith("sqlite+libsql:/") and not embedded_dsn.startswith("sqlite+libsql://"):
        embedded_dsn = "sqlite+libsql:///" + embedded_dsn.split(":/", 1)[1].lstrip("/")
    if not embedded_dsn.startswith("sqlite+libsql:///"):
        return  # not an embedded DSN; nothing to return to
    if "sync_url=libsql://" in embedded_dsn:
        embedded_dsn = embedded_dsn.replace("sync_url=libsql://", "sync_url=https://")
    if "secure=" in embedded_dsn.lower():
        embedded_dsn = embedded_dsn.replace("&secure=true", "").replace("?secure=true", "?").replace("?&", "?")
        if embedded_dsn.endswith("?"):
            embedded_dsn = embedded_dsn[:-1]

    token = str(_get_secret("TURSO_AUTH_TOKEN") or "")
    if not token:
        return

    try:
        e2 = create_engine(
            embedded_dsn,
            connect_args={"auth_token": token},
            pool_pre_ping=True,
            pool_recycle=180,
        )
        with e2.connect() as c:
            c.exec_driver_sql("SELECT 1")
            row = c.exec_driver_sql(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                ("vendors",),
            ).fetchone()
            if not row:
                return

        # Swap back to embedded
        engine = e2
        engine_info.update(
            {
                "using_remote": True,
                "sqlalchemy_url": embedded_dsn,
                "dialect": "sqlite",
                "driver": "libsql",
                "strategy": "embedded_replica",
            }
        )

        if not st.session_state.get("ro_returned_info"):
            st.session_state["ro_returned_info"] = True
            st.info("Replica is ready; switched back to embedded.")

        try:
            st.cache_data.clear()
        except Exception:
            pass

    except Exception:
        # If anything fails, stay on failover; try again next run
        pass


# Run warm-up; if schema still missing, fail over to direct remote, then attempt to return later
if not _warmup_embedded_replica():
    _failover_to_direct_remote_if_needed()
_maybe_return_to_embedded()


# -----------------------------
# Data helpers
# -----------------------------
@st.cache_data(ttl=60)
def fetch_df(sql: str, params: Dict | None = None) -> pd.DataFrame:
    # Use the cached engine from get_engine_and_info
    try:
        with engine.connect() as conn:
            if params:
                # NOTE: exec_driver_sql with dict params can error on libsql; don't pass params here.
                result = conn.exec_driver_sql(sql, params)
            else:
                result = conn.exec_driver_sql(sql)
            rows = result.fetchall()
            cols = result.keys()
        return pd.DataFrame(rows, columns=list(cols))
    except Exception as ex:
        st.error(f"DB query failed: {type(ex).__name__}: {ex}")
        raise


def vendors_df() -> pd.DataFrame:
    # Case-insensitive ordering without calling lower()
    sql = (
        "SELECT id, business_name, category, service, contact_name, phone, address, website, notes, keywords "
        "FROM vendors ORDER BY business_name COLLATE NOCASE"
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
    components.html(html_table, height=520, scrolling=True)


# -----------------------------
# Main UI
# -----------------------------
def main():
    if HELP_MD:
        with st.expander(HELP_TITLE or "Provider Help / Tips", expanded=False):
            st.markdown(HELP_MD)

    q = st.text_input("Search (partial match across most fields)", placeholder="e.g., plumb or 210-555-…")

    df = vendors_df()
    if HIDE_ID and "id" in df.columns:
        df = df.drop(columns=["id"])

    if q:
        ql = q.lower().strip()
        mask = pd.Series(False, index=df.index)
        for col in [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("string")]:
            mask = mask | df[col].astype(str).str.lower().str.contains(ql, na=False)
        df = df[mask].copy()

    render_html_table(df, sticky_first_col=STICKY_FIRST)

    # Downloads
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download filtered as providers.csv",
            df.to_csv(index=False).encode("utf-8"),
            "providers.csv",
            "text/csv",
        )
    with c2:
        full = vendors_df()
        if HIDE_ID and "id" in full.columns:
            full = full.drop(columns=["id"])
        st.download_button(
            "Download FULL export (all rows)",
            full.to_csv(index=False).encode("utf-8"),
            "providers_full.csv",
            "text/csv",
        )

    # Debug / Maintenance (safe for end users; no write ops against DB)
    if st.button("Show Debug / Status", key="dbg_btn_ro"):
        st.write("Status & Secrets (debug)")
        st.json(engine_info)

        # Library versions
        st.json(
            {
                "streamlit": st.__version__,
                "pandas": pd.__version__,
                "SQLAlchemy": __import__("sqlalchemy").__version__,
                "sqlalchemy-libsql": sqlalchemy_libsql.__version__,
            }
        )

        # Raw values (length only for token)
        src_secrets_dsn = st.secrets.get("TURSO_DATABASE_URL")
        src_secrets_tok = st.secrets.get("TURSO_AUTH_TOKEN")
        env_dsn = os.environ.get("TURSO_DATABASE_URL")
        env_tok = os.environ.get("TURSO_AUTH_TOKEN")

        def _repr(v):
            return repr(v) if v is not None else "None"

        st.caption("Raw values as seen by the app (repr):")
        st.json(
            {
                "secrets.TURSO_DATABASE_URL": _repr(src_secrets_dsn),
                "secrets.TURSO_AUTH_TOKEN": f"<len {len(src_secrets_tok) if src_secrets_tok else 0}>",
                "env.TURSO_DATABASE_URL": _repr(env_dsn),
                "env.TURSO_AUTH_TOKEN": f"<len {len(env_tok) if env_tok else 0}>",
            }
        )

        # Embedded file status
        url = str(engine_info.get("sqlalchemy_url", ""))
        if url.startswith("sqlite+libsql:///"):
            file_part = url.split("sqlite+libsql:///", 1)[1].split("?", 1)[0]
            embed_path = "/" + file_part.lstrip("/")
            st.write("Embedded file status")
            if os.path.exists(embed_path):
                st.write(f"• {embed_path} — {os.path.getsize(embed_path)} bytes; mtime={time.ctime(os.path.getmtime(embed_path))}")
            else:
                st.write(f"• {embed_path} — (not found)")
        else:
            st.write("Embedded file status")
            st.write("• Not using embedded DSN (serving from remote host).")

        # Schema probe
        try:
            with engine.connect() as conn:
                names = [r[0] for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()]
            st.write("Schema probe")
            st.write("Tables:")
            st.write(names)
            if "vendors" not in names:
                st.warning("Table 'vendors' not found in this database.")
        except Exception as e:
            st.error(f"Probe failed: {e}")

    # Maintenance (optional UI toggle)
    maint = str(_get_secret("READONLY_MAINTENANCE_ENABLE", "0")).lower() in ("1", "true", "yes")
    if maint:
        st.markdown("### Maintenance (embedded replica)")
        st.caption("Only enabled when READONLY_MAINTENANCE_ENABLE=1 in secrets. Use carefully.")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Delete embedded file (force re-sync)", key="del_embedded"):
                try:
                    # Derive absolute path from current DSN
                    url = str(engine_info.get("sqlalchemy_url", ""))
                    if not url.startswith("sqlite+libsql:///"):
                        st.info("Not using embedded DSN; nothing to delete.")
                    else:
                        file_part = url.split("sqlite+libsql:///", 1)[1].split("?", 1)[0]
                        embed_path = "/" + file_part.lstrip("/")
                        os.remove(embed_path)
                        st.success(f"Deleted {embed_path}. Reload the app to re-sync from Turso.")
                except FileNotFoundError:
                    st.info("No embedded file found.")
                except Exception as ex:
                    st.error(f"Delete failed: {type(ex).__name__}: {ex}")
        with col_b:
            if st.button("Clear data cache", key="clear_cache_ro"):
                try:
                    st.cache_data.clear()
                    st.success("Cache cleared.")
                except Exception as ex:
                    st.error(f"Cache clear failed: {ex}")


if __name__ == "__main__":
    main()
