# -*- coding: utf-8 -*-
# app_admin.py - Providers Admin (direct remote recommended; same engine rules)
from __future__ import annotations

import os
from typing import Dict, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import sqlalchemy_libsql  # registers 'sqlite+libsql' dialect entrypoint


# -----------------------------
# Page config
# -----------------------------
PAGE_TITLE = os.environ.get("ADMIN_PAGE_TITLE", "Providers Admin")
st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state="expanded")


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


ALLOW_SQLITE_FALLBACK = str(_get_secret("ALLOW_SQLITE_FALLBACK", "0")).lower() in ("1", "true", "yes")


# -----------------------------
# Engine builder (shared conventions)
# -----------------------------
def _normalize_dsn(raw: str) -> tuple[str, bool]:
    dsn = raw.strip()
    if dsn.startswith("libsql://"):
        dsn = "sqlite+libsql://" + dsn.split("://", 1)[1]
    if dsn.startswith("sqlite+libsql:/") and not dsn.startswith("sqlite+libsql://"):
        dsn = "sqlite+libsql:///" + dsn.split(":/", 1)[1].lstrip("/")
    is_embedded = dsn.startswith("sqlite+libsql:///")
    if is_embedded:
        if "sync_url=libsql://" in dsn:
            dsn = dsn.replace("sync_url=libsql://", "sync_url=https://")
        if "secure=" in dsn.lower():
            dsn = dsn.replace("&secure=true", "").replace("?secure=true", "?").replace("?&", "?")
            if dsn.endswith("?"):
                dsn = dsn[:-1]
    else:
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


def build_engine() -> Tuple[Engine, Dict[str, str]]:
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
        with e.connect() as c:
            c.exec_driver_sql("SELECT 1")
        info: Dict[str, str] = {
            "using_remote": True,
            "dialect": e.dialect.name,
            "driver": getattr(e.dialect, "driver", ""),
            "sqlalchemy_url": dsn,
            "sync_url": dsn,
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
def get_engine_and_info() -> Tuple[Engine, Dict[str, str]]:
    return build_engine()


engine, engine_info = get_engine_and_info()


# -----------------------------
# Admin gate
# -----------------------------
ADMIN_PASSWORD = _get_secret("ADMIN_PASSWORD", "")
DISABLE_ADMIN_PASSWORD = str(_get_secret("DISABLE_ADMIN_PASSWORD", "0")).lower() in ("1", "true", "yes")

def _check_admin() -> bool:
    if DISABLE_ADMIN_PASSWORD:
        return True
    with st.form("admin_login", clear_on_submit=False):
        pw = st.text_input("Admin password", type="password")
        ok = st.form_submit_button("Enter")
    if ok and pw == ADMIN_PASSWORD and pw != "":
        st.session_state["is_admin"] = True
    return bool(st.session_state.get("is_admin"))

if not _check_admin():
    st.stop()


# ----------------
