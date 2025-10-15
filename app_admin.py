# ==== BEGIN: FILE TOP (imports + page_config + Early Boot) ====
# -*- coding: utf-8 -*-
from __future__ import annotations

# -------- Imports (single source of truth) --------
import os
import sys
import time
import re
import hmac
import uuid
import platform
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Iterable

import pandas as pd
import requests
import sqlalchemy
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# ---- Streamlit page config MUST be first Streamlit call ----
st.set_page_config(page_title="Vendors Admin", layout="wide", initial_sidebar_state="expanded")

# ==== BEGIN: Admin boot bundle (version banner + guards + engine + init) ====

# --- Unified libsql version + details (authoritative from package metadata) ---
import importlib
import importlib.metadata as _im

_lib_file = "n/a"
try:
    _lib_ver_display = _im.version("sqlalchemy-libsql")
except Exception:
    try:
        import sqlalchemy_libsql as _lib_mod
        _lib_ver_display = getattr(_lib_mod, "__version__", "unknown")
    except Exception:
        _lib_ver_display = "unknown"

try:
    _lib_mod = importlib.import_module("sqlalchemy_libsql")
    _lib_file = getattr(_lib_mod, "__file__", "n/a")
except Exception:
    pass

# --- Streamlit run-context guard (prevents SessionInfo errors during import) ---
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx
except Exception:
    _get_ctx = None

def _has_streamlit_ctx() -> bool:
    try:
        return (_get_ctx() is not None) if _get_ctx else False
    except Exception:
        return False

# --- ENV + SHOW_DEBUG derivation (prod hides debug by default) ---
_ENV = str(st.secrets.get("ENV", "prod")).strip().lower()
_raw_show_debug = st.secrets.get("SHOW_DEBUG", "")
if isinstance(_raw_show_debug, bool):
    _SHOW_DEBUG = _raw_show_debug
else:
    _SHOW_DEBUG = (_ENV != "prod")

# --- Version banner (sidebar, debug-only; hidden in prod) ---
if _SHOW_DEBUG and _has_streamlit_ctx() and bool(st.secrets.get("SHOW_STATUS", True)):
    try:
        st.sidebar.info(
            "Versions | "
            f"py {sys.version.split()[0]} | "
            f"streamlit {st.__version__} | "
            f"pandas {pd.__version__} | "
            f"SA {sqlalchemy.__version__} | "
            f"libsql {_lib_ver_display} | "
            f"requests {requests.__version__}"
        )
        with st.sidebar.expander("libsql details"):
            st.write({"module_file": _lib_file, "pkg_version": _lib_ver_display})
    except Exception as _e:
        # Only warn in debug to avoid noise/headless issues
        st.sidebar.warning(f"Version banner failed: {_e}")
# ==== END: FILE TOP (imports + page_config + Early Boot) ====

# --- Secrets / strategy (validated) ---
_DB_STRATEGY   = str(st.secrets.get("DB_STRATEGY", "embedded_replica")).strip().lower()
_TURSO_URL     = str(st.secrets.get("TURSO_DATABASE_URL", "")).strip()
_TURSO_TOKEN   = str(st.secrets.get("TURSO_AUTH_TOKEN", "")).strip()
_EMBEDDED_PATH = str(st.secrets.get("EMBEDDED_DB_PATH", "/mount/src/providers/vendors-embedded.db")).strip()

def _validate_sync_url(u: str) -> str:
    if not u:
        raise ValueError("TURSO_DATABASE_URL is empty; expected libsql://<host>")
    if u.startswith("sqlite+libsql://"):
        raise ValueError("TURSO_DATABASE_URL must start with libsql:// (not sqlite+libsql://)")
    scheme = u.split("://", 1)[0].lower()
    if scheme != "libsql":
        raise ValueError(f"Unsupported sync URL scheme: {scheme}:// (expected libsql://)")
    return u

# Require remote secrets only when syncing
if _DB_STRATEGY in ("embedded_replica", "replica", "sync"):
    try:
        _validate_sync_url(_TURSO_URL)
    except Exception as _e:
        st.error(f"Bad TURSO_DATABASE_URL: {type(_e).__name__}: {_e}")
        st.stop()
    if not _TURSO_TOKEN:
        st.error("Missing required secret: TURSO_AUTH_TOKEN")
        st.stop()

# Back-compat for any remaining references
_strategy = _DB_STRATEGY

# --- URL masker: never leak tokens/paths; show scheme://host only ---
from urllib.parse import urlparse
def _mask_sync_url(u: str) -> str:
    if not u:
        return ""
    try:
        p = urlparse(u)
        host = p.hostname or ""
        return (p.scheme or "") + "://" + host
    except Exception:
        return (u.split("://", 1)[0] + "://") if "://" in u else ""

# --- Canonical engine builder (embedded replica with libsql sync) ---
def build_engine_and_probe() -> tuple[Engine, dict]:
    engine_url = f"sqlite+libsql:///{_EMBEDDED_PATH}"

    # Diagnostics payload (use unified libsql version)
    dbg = {
        "host": platform.node() or "localhost",
        "strategy": _DB_STRATEGY,
        "python": sys.version.split()[0],
        "sqlalchemy_url": engine_url,
        "embedded_path": _EMBEDDED_PATH,
        "libsql_ver": _lib_ver_display,           # unified version string
        "sync_url_scheme": "",
    }

    connect_args: Dict[str, Any] = {}

    if _DB_STRATEGY in ("embedded_replica", "replica", "sync"):
        valid_sync = _validate_sync_url(_TURSO_URL)
        dbg["sync_url_scheme"] = _mask_sync_url(valid_sync).split("://", 1)[0] + "://"
        connect_args["sync_url"] = valid_sync
        if _TURSO_TOKEN:
            connect_args["auth_token"] = _TURSO_TOKEN

    elif _DB_STRATEGY == "remote_only":
        engine_url = _validate_sync_url(_TURSO_URL)
        dbg["sqlalchemy_url"] = engine_url
        dbg["embedded_path"] = ""
        dbg["sync_url_scheme"] = "libsql://"
        if _TURSO_TOKEN:
            connect_args["auth_token"] = _TURSO_TOKEN

    # else: embedded_only => no sync_url

    eng = create_engine(engine_url, connect_args=connect_args)

    # Quick sanity (fail fast if driver/URL is wrong)
    with eng.connect() as cx:
        cx.exec_driver_sql("PRAGMA journal_mode")

    return eng, dbg

# --- Single guarded init + diagnostics (no SessionInfo errors) ---
try:
    ENGINE, _DB_DBG = build_engine_and_probe()

    if _has_streamlit_ctx():
        # Sidebar notice only when debugging (keeps sidebar hidden in prod)
        if _SHOW_DEBUG:
            st.sidebar.success("DB ready")

        # Optional: boot diagnostics + quick COUNT (debug-only)
        if _SHOW_DEBUG:
            with st.expander("Boot diagnostics (ENGINE + secrets)"):
                st.json(_DB_DBG)

            if bool(st.secrets.get("SHOW_COUNT", True)):
                try:
                    with ENGINE.connect() as cx:
                        cnt = cx.exec_driver_sql("SELECT COUNT(*) FROM vendors").scalar()
                    st.info(f"Vendors table row count: {int(cnt or 0)}")
                except Exception as _e:
                    st.warning(f"Quick vendors count failed: {type(_e).__name__}: {_e}")

        # Success marker only when debugging
        if _SHOW_DEBUG:
            st.success("App reached post-boot marker ✅")

        # Stash for reuse (UI context only)
        st.session_state["ENGINE"] = ENGINE
        st.session_state["DB_DBG"]  = _DB_DBG

        if _SHOW_DEBUG:
            with st.expander("Boot diagnostics (ENGINE + secrets)"):
                st.json(_DB_DBG)

            if bool(st.secrets.get("SHOW_COUNT", True)):
                try:
                    with ENGINE.connect() as cx:
                        cnt = cx.exec_driver_sql("SELECT COUNT(*) FROM vendors").scalar()
                    st.info(f"Vendors table row count: {int(cnt or 0)}")
                except Exception as _e:
                    st.warning(f"Quick vendors count failed: {type(_e).__name__}: {_e}")

        # Success banner hidden in prod; show only when debugging
        if _SHOW_DEBUG:
            st.success("App reached post-boot marker ✅")

        # Stash for reuse (UI context only)
        st.session_state["ENGINE"] = ENGINE
        st.session_state["DB_DBG"] = _DB_DBG
    else:
        # Headless import path: validate engine without touching UI/session
        with ENGINE.connect() as cx:
            cx.exec_driver_sql("SELECT 1")

except Exception as e:
    if _has_streamlit_ctx():
        st.error(f"Database init failed: {e.__class__.__name__}: {e}")
        st.stop()
    else:
        # Re-raise in headless contexts so CI/linters fail loud
        raise
# --- End: Single guarded init + diagnostics ---


# ==== END: Admin boot bundle (version banner + guards + engine + init) ====

# ==== END: FILE TOP (imports + page_config + Early Boot) ====

# -----------------------------
# Helpers
# -----------------------------
def _as_bool(v, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")

def _get_secret(name: str, default: str | None = None) -> str | None:
    """Prefer Streamlit secrets, fallback to environment, then default."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

# Deterministic resolution (secrets → env → code default)
def _resolve_bool(name: str, code_default: bool) -> bool:
    v = _get_secret(name, None)
    return _as_bool(v, default=code_default)

def _resolve_str(name: str, code_default: str | None) -> str | None:
    v = _get_secret(name, None)
    return v if v is not None else code_default

def _ct_equals(a: str, b: str) -> bool:
    """Constant-time string compare for secrets."""
    return hmac.compare_digest((a or ""), (b or ""))


# -----------------------------
# Hrana/libSQL transient error retry
# -----------------------------
def _is_hrana_stale_stream_error(err: Exception) -> bool:
    s = str(err).lower()
    return ("hrana" in s and "404" in s and "stream not found" in s) or ("stream not found" in s)

def _exec_with_retry(engine: Engine, sql: str, params: Dict | None = None, *, tries: int = 2):
    """
    Execute a write (INSERT/UPDATE/DELETE) with a one-time retry on Hrana 'stream not found'.
    Returns the result proxy so you can read .rowcount.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            with engine.begin() as conn:
                return conn.execute(sql_text(sql), params or {})
        except Exception as e:
            if attempt < tries and _is_hrana_stale_stream_error(e):
                try:
                    engine.dispose()  # drop pooled connections
                except Exception:
                    pass
                time.sleep(0.2)
                continue
            raise

def _fetch_with_retry(engine: Engine, sql: str, params: Dict | None = None, *, tries: int = 2) -> pd.DataFrame:
    """
    Execute a read (SELECT) with a one-time retry on Hrana 'stream not found'.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            with engine.connect() as conn:
                res = conn.execute(sql_text(sql), params or {})
                return pd.DataFrame(res.mappings().all())
        except Exception as e:
            if attempt < tries and _is_hrana_stale_stream_error(e):
                try:
                    engine.dispose()
                except Exception:
                    pass
                time.sleep(0.2)
                continue
            raise


# ---------- Form state helpers (Add / Edit / Delete) ----------
# Add form keys
ADD_FORM_KEYS = [
    "add_business_name", "add_category", "add_service", "add_contact_name",
    "add_phone", "add_address", "add_website", "add_notes", "add_keywords",
]

def _init_add_form_defaults():
    for k in ADD_FORM_KEYS:
        if k not in st.session_state:
            st.session_state[k] = ""
    st.session_state.setdefault("add_form_version", 0)
    st.session_state.setdefault("_pending_add_reset", False)
    st.session_state.setdefault("add_last_done", None)
    st.session_state.setdefault("add_nonce", uuid.uuid4().hex)

def _apply_add_reset_if_needed():
    """Apply queued reset BEFORE rendering widgets to avoid invalid-option errors."""
    if st.session_state.get("_pending_add_reset"):
        for k in ADD_FORM_KEYS:
            st.session_state[k] = ""
        st.session_state["_pending_add_reset"] = False
        st.session_state["add_form_version"] += 1

def _queue_add_form_reset():
    st.session_state["_pending_add_form_reset"] = True
    st.session_state["_pending_add_reset"] = True  # keep original flag for compatibility

# Edit form keys
EDIT_FORM_KEYS = [
    "edit_vendor_id", "edit_business_name", "edit_category", "edit_service",
    "edit_contact_name", "edit_phone", "edit_address", "edit_website",
    "edit_notes", "edit_keywords", "edit_row_updated_at", "edit_last_loaded_id",
]

def _init_edit_form_defaults():
    defaults = {
        "edit_vendor_id": None,
        "edit_business_name": "",
        "edit_category": "",
        "edit_service": "",
        "edit_contact_name": "",
        "edit_phone": "",
        "edit_address": "",
        "edit_website": "",
        "edit_notes": "",
        "edit_keywords": "",
        "edit_row_updated_at": None,
        "edit_last_loaded_id": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("edit_form_version", 0)
    st.session_state.setdefault("_pending_edit_reset", False)
    st.session_state.setdefault("edit_last_done", None)
    st.session_state.setdefault("edit_nonce", uuid.uuid4().hex)

def _apply_edit_reset_if_needed():
    """
    Apply queued reset BEFORE rendering edit widgets.
    Also clear the selection (edit_vendor_id) and the selectbox key so the UI returns to “— Select —”.
    """
    if st.session_state.get("_pending_edit_reset"):
        for k in EDIT_FORM_KEYS:
            if k == "edit_vendor_id":
                st.session_state[k] = None
            elif k in ("edit_row_updated_at", "edit_last_loaded_id"):
                st.session_state[k] = None
            else:
                st.session_state[k] = ""
        if "edit_provider_label" in st.session_state:
            del st.session_state["edit_provider_label"]
        st.session_state["_pending_edit_reset"] = False
        st.session_state["edit_form_version"] += 1

def _queue_edit_form_reset():
    st.session_state["_pending_edit_reset"] = True

# Delete form keys
DELETE_FORM_KEYS = ["delete_vendor_id"]

def _init_delete_form_defaults():
    st.session_state.setdefault("delete_vendor_id", None)
    st.session_state.setdefault("delete_form_version", 0)
    st.session_state.setdefault("_pending_delete_reset", False)
    st.session_state.setdefault("delete_last_done", None)
    st.session_state.setdefault("delete_nonce", uuid.uuid4().hex)

def _apply_delete_reset_if_needed():
    if st.session_state.get("_pending_delete_reset"):
        st.session_state["delete_vendor_id"] = None
        if "delete_provider_label" in st.session_state:
            del st.session_state["delete_provider_label"]
        st.session_state["_pending_delete_reset"] = False
        st.session_state["delete_form_version"] += 1

def _queue_delete_form_reset():
    st.session_state["_pending_delete_reset"] = True

# Nonce helpers
def _nonce(name: str) -> str:
    return st.session_state.get(f"{name}_nonce")

def _nonce_rotate(name: str) -> None:
    st.session_state[f"{name}_nonce"] = uuid.uuid4().hex

# General-purpose key helpers (used in Category/Service admins)
def _clear_keys(*keys: str) -> None:
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]

def _set_empty(*keys: str) -> None:
    for k in keys:
        st.session_state[k] = ""

def _reset_select(key: str, sentinel: str = "— Select —") -> None:
    st.session_state[key] = sentinel


# ---------- Category / Service queued reset helpers ----------
def _init_cat_defaults():
    st.session_state.setdefault("cat_form_version", 0)
    st.session_state.setdefault("_pending_cat_reset", False)

def _apply_cat_reset_if_needed():
    if st.session_state.get("_pending_cat_reset"):
        st.session_state["cat_add"] = ""
        st.session_state["cat_rename"] = ""
        for k in ("cat_old", "cat_del", "cat_reassign_to"):
            if k in st.session_state:
                del st.session_state[k]
        st.session_state["_pending_cat_reset"] = False
        st.session_state["cat_form_version"] += 1

def _queue_cat_reset():
    st.session_state["_pending_cat_reset"] = True

def _init_svc_defaults():
    st.session_state.setdefault("svc_form_version", 0)
    st.session_state.setdefault("_pending_svc_reset", False)

def _apply_svc_reset_if_needed():
    if st.session_state.get("_pending_svc_reset"):
        st.session_state["svc_add"] = ""
        st.session_state["svc_rename"] = ""
        for k in ("svc_old", "svc_del", "svc_reassign_to"):
            if k in st.session_state:
                del st.session_state[k]
        st.session_state["_pending_svc_reset"] = False
        st.session_state["svc_form_version"] += 1

def _queue_svc_reset():
    st.session_state["_pending_svc_reset"] = True


# -----------------------------
# Page CSS (no second page_config here)
# -----------------------------
LEFT_PAD_PX = int(_resolve_str("page_left_padding_px", "40") or "40")

st.markdown(
    f"""
    <style>
      [data-testid="stAppViewContainer"] .main .block-container {{
        padding-left: {LEFT_PAD_PX}px !important;
        padding-right: 0 !important;
      }}
      div[data-testid="stDataFrame"] table {{ white-space: nowrap; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# DB helpers (schema + IO) — plus cascade ops for category/service changes
# -----------------------------
REQUIRED_VENDOR_COLUMNS: List[str] = ["business_name", "category"]  # service optional

def ensure_schema(engine: Engine) -> None:
    """
    Create tables if missing (hard fail with exact SQL on error).
    Create indexes best-effort (warn, don't crash).
    Optional ALTERs (migrations) also best-effort.
    """
    table_ddls = [
        # vendors
        """
        CREATE TABLE IF NOT EXISTS vendors (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          category TEXT NOT NULL,
          service TEXT,
          business_name TEXT NOT NULL,
          contact_name TEXT,
          phone TEXT,
          address TEXT,
          website TEXT,
          notes TEXT,
          keywords TEXT,
          created_at TEXT,
          updated_at TEXT,
          updated_by TEXT
        )
        """,
        # categories
        """
        CREATE TABLE IF NOT EXISTS categories (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT UNIQUE
        )
        """,
        # services
        """
        CREATE TABLE IF NOT EXISTS services (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT UNIQUE
        )
        """,
    ]

    # Note: include a plain index on service to speed cascades
    index_ddls = [
        "CREATE INDEX IF NOT EXISTS idx_vendors_cat ON vendors(category)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_bus ON vendors(business_name)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_kw  ON vendors(keywords)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_bus_lower ON vendors(lower(business_name))",
        "CREATE INDEX IF NOT EXISTS idx_vendors_cat_lower ON vendors(lower(category))",
        "CREATE INDEX IF NOT EXISTS idx_vendors_svc_lower ON vendors(lower(service))",
        "CREATE INDEX IF NOT EXISTS idx_vendors_svc ON vendors(service)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_phone ON vendors(phone)",
    ]

    with engine.begin() as conn:
        # --- 1) Tables: hard requirement (raise with exact statement) ---
        for s in table_ddls:
            stmt = s.strip()
            try:
                conn.exec_driver_sql(stmt)
            except Exception as e:
                raise ValueError(
                    f"TABLE DDL failed: {e.__class__.__name__}: {e}\n--- SQL ---\n{stmt}\n"
                ) from e

        # --- 2) Indexes: best-effort (warn, do not abort app) ---
        for s in index_ddls:
            stmt = s.strip()
            try:
                conn.exec_driver_sql(stmt)
            except Exception as e:
                if '_SHOW_DEBUG' in globals() and _SHOW_DEBUG and '_has_streamlit_ctx' in globals() and _has_streamlit_ctx():
                    st.warning(f"Index DDL skipped: {e.__class__.__name__}: {e}\n— {stmt}")

        # --- 3) Optional migrations: add missing columns if ever needed (soft) ---
        def _table_columns(table: str) -> set[str]:
            rows = conn.execute(sql_text(f"PRAGMA table_info({table})")).fetchall()
            return {str(r[1]) for r in rows}

        def _add_column_if_missing(table: str, decl: str) -> None:
            # decl like: "computed_keywords TEXT"
            col = decl.split()[0]
            try:
                if col not in _table_columns(table):
                    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {decl}")
            except Exception as e:
                if '_SHOW_DEBUG' in globals() and _SHOW_DEBUG and '_has_streamlit_ctx' in globals() and _has_streamlit_ctx():
                    st.warning(f"ALTER skipped: {table}.{col}: {e.__class__.__name__}: {e}")
        # Example (disabled):
        # _add_column_if_missing("vendors", "computed_keywords TEXT")


def _normalize_phone(val: str | None) -> str:
    if not val:
        return ""
    digits = re.sub(r"\D", "", str(val))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits if len(digits) == 10 else digits


def _format_phone(val: str | None) -> str:
    s = re.sub(r"\D", "", str(val or ""))
    if len(s) == 10:
        return f"({s[0:3]}) {s[3:6]}-{s[6:10]}"
    return (val or "").strip()


def _sanitize_url(url: str | None) -> str:
    if not url:
        return ""
    url = url.strip()
    if url and not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return url


def load_df(engine: Engine) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(sql_text("SELECT * FROM vendors ORDER BY lower(business_name)"), conn)

    for col in [
        "contact_name",
        "phone",
        "address",
        "website",
        "notes",
        "keywords",
        "service",
        "created_at",
        "updated_at",
        "updated_by",
    ]:
        if col not in df.columns:
            df[col] = ""

    df["phone_fmt"] = df["phone"].apply(_format_phone)
    return df


def list_names(engine: Engine, table: str) -> list[str]:
    with engine.begin() as conn:
        rows = conn.execute(sql_text(f"SELECT name FROM {table} ORDER BY lower(name)")).fetchall()
    return [r[0] for r in rows]


def usage_count(engine: Engine, col: str, name: str) -> int:
    with engine.begin() as conn:
        cnt = conn.execute(sql_text(f"SELECT COUNT(*) FROM vendors WHERE {col} = :n"), {"n": name}).scalar()
    return int(cnt or 0)


# ---------- Cascade helpers ----------
def rename_category_and_cascade(engine: Engine, old: str, new: str) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("INSERT OR IGNORE INTO categories(name) VALUES(:n)"), {"n": new})
        conn.execute(sql_text("UPDATE vendors SET category=:new WHERE category=:old"), {"new": new, "old": old})
        conn.execute(sql_text("DELETE FROM categories WHERE name=:old"), {"old": old})

def delete_category_with_reassign(engine: Engine, tgt: str, repl: str) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("UPDATE vendors SET category=:repl WHERE category=:tgt"), {"repl": repl, "tgt": tgt})
        conn.execute(sql_text("DELETE FROM categories WHERE name=:tgt"), {"tgt": tgt})

def rename_service_and_cascade(engine: Engine, old: str, new: str) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("INSERT OR IGNORE INTO services(name) VALUES(:n)"), {"n": new})
        conn.execute(sql_text("UPDATE vendors SET service=:new WHERE service=:old"), {"new": new, "old": old})
        conn.execute(sql_text("DELETE FROM services WHERE name=:old"), {"old": old})

def delete_service_with_reassign(engine: Engine, tgt: str, repl: str) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("UPDATE vendors SET service=:repl WHERE service=:tgt"), {"repl": repl, "tgt": tgt})
        conn.execute(sql_text("DELETE FROM services WHERE name=:tgt"), {"tgt": tgt})


# ---------- Category Admin
with _tabs[2]:
    st.caption("Category is required. Manage the reference list and reassign vendors safely.")
    _init_cat_defaults()
    _apply_cat_reset_if_needed()

    cats = list_names(engine, "categories")
    cat_opts = ["— Select —"] + cats  # sentinel first

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Add Category")
        new_cat = st.text_input("New category name", key="cat_add")
        if st.button("Add Category", key="cat_add_btn"):
            if not (new_cat or "").strip():
                st.error("Enter a name.")
            else:
                try:
                    _exec_with_retry(engine, "INSERT OR IGNORE INTO categories(name) VALUES(:n)", {"n": new_cat.strip()})
                    st.success("Added (or already existed).")
                    _queue_cat_reset()
                    st.rerun()
                except Exception as e:
                    st.error(f"Add category failed: {e}")

        st.subheader("Rename Category")
        if cats:
            old = st.selectbox("Current", options=cat_opts, key="cat_old")  # no index
            new = st.text_input("New name", key="cat_rename")
            if st.button("Rename", key="cat_rename_btn"):
                if old == "— Select —":
                    st.error("Pick a category to rename.")
                elif not (new or "").strip():
                    st.error("Enter a new name.")
                else:
                    try:
                        rename_category_and_cascade(engine, old, new.strip())
                        st.success("Renamed and reassigned.")
                        _queue_cat_reset()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Rename category failed: {e}")

    with colB:
        st.subheader("Delete / Reassign")
        if cats:
            tgt = st.selectbox("Category to delete", options=cat_opts, key="cat_del")  # no index
            if tgt == "— Select —":
                st.write("Select a category.")
            else:
                cnt = usage_count(engine, "category", tgt)
                st.write(f"In use by {cnt} vendor(s).")
                if cnt == 0:
                    if st.button("Delete category (no usage)", key="cat_del_btn"):
                        try:
                            _exec_with_retry(engine, "DELETE FROM categories WHERE name=:n", {"n": tgt})
                            st.success("Deleted.")
                            _queue_cat_reset()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete category failed: {e}")
                else:
                    repl_options = ["— Select —"] + [c for c in cats if c != tgt]
                    repl = st.selectbox("Reassign vendors to…", options=repl_options, key="cat_reassign_to")
                    if st.button("Reassign vendors then delete", key="cat_reassign_btn"):
                        if repl == "— Select —":
                            st.error("Choose a category to reassign to.")
                        else:
                            try:
                                delete_category_with_reassign(engine, tgt, repl)
                                st.success("Reassigned and deleted.")
                                _queue_cat_reset()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Reassign+delete failed: {e}")

# ---------- Service Admin
with _tabs[3]:
    st.caption("Service is optional on vendors. Manage the reference list here.")
    _init_svc_defaults()
    _apply_svc_reset_if_needed()

    servs = list_names(engine, "services")
    svc_opts = ["— Select —"] + servs  # sentinel first

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Add Service")
        new_s = st.text_input("New service name", key="svc_add")
        if st.button("Add Service", key="svc_add_btn"):
            if not (new_s or "").strip():
                st.error("Enter a name.")
            else:
                try:
                    _exec_with_retry(engine, "INSERT OR IGNORE INTO services(name) VALUES(:n)", {"n": new_s.strip()})
                    st.success("Added (or already existed).")
                    _queue_svc_reset()
                    st.rerun()
                except Exception as e:
                    st.error(f"Add service failed: {e}")

        st.subheader("Rename Service")
        if servs:
            old = st.selectbox("Current", options=svc_opts, key="svc_old")  # no index
            new = st.text_input("New name", key="svc_rename")
            if st.button("Rename Service", key="svc_rename_btn"):
                if old == "— Select —":
                    st.error("Pick a service to rename.")
                elif not (new or "").strip():
                    st.error("Enter a new name.")
                else:
                    try:
                        rename_service_and_cascade(engine, old, new.strip())
                        st.success(f"Renamed service: {old} → {new.strip()}")
                        _queue_svc_reset()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Rename service failed: {e}")

    with colB:
        st.subheader("Delete / Reassign")
        if servs:
            tgt = st.selectbox("Service to delete", options=svc_opts, key="svc_del")  # no index
            if tgt == "— Select —":
                st.write("Select a service.")
            else:
                cnt = usage_count(engine, "service", tgt)
                st.write(f"In use by {cnt} vendor(s).")
                if cnt == 0:
                    if st.button("Delete service (no usage)", key="svc_del_btn"):
                        try:
                            _exec_with_retry(engine, "DELETE FROM services WHERE name=:n", {"n": tgt})
                            st.success("Deleted.")
                            _queue_svc_reset()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete service failed: {e}")
                else:
                    repl_options = ["— Select —"] + [s for s in servs if s != tgt]
                    repl = st.selectbox("Reassign vendors to…", options=repl_options, key="svc_reassign_to")
                    if st.button("Reassign vendors then delete service", key="svc_reassign_btn"):
                        if repl == "— Select —":
                            st.error("Choose a service to reassign to.")
                        else:
                            try:
                                delete_service_with_reassign(engine, tgt, repl)
                                st.success("Reassigned and deleted.")
                                _queue_svc_reset()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Reassign+delete service failed: {e}")

# ---------- Maintenance
with _tabs[4]:
    st.caption("One-click cleanups for legacy data.")

    st.subheader("Export / Import")

    # Export full, untruncated CSV of all columns/rows
    query = "SELECT * FROM vendors ORDER BY lower(business_name)"
    with engine.begin() as conn:
        full = pd.read_sql(sql_text(query), conn)

    # Dual exports: full dataset — formatted phones and digits-only
    full_formatted = full.copy()

    def _format_phone_digits(x: str | int | None) -> str:
        s = re.sub(r"\D+", "", str(x or ""))
        return f"({s[0:3]}) {s[3:6]}-{s[6:10]}" if len(s) == 10 else s

    if "phone" in full_formatted.columns:
        full_formatted["phone"] = full_formatted["phone"].apply(_format_phone_digits)

    colA, colB = st.columns([1, 1])
    with colA:
        st.download_button(
            "Export all vendors (formatted phones)",
            data=full_formatted.to_csv(index=False).encode("utf-8"),
            file_name=f"providers_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.csv",
            mime="text/csv",
        )
    with colB:
        st.download_button(
            "Export all vendors (digits-only phones)",
            data=full.to_csv(index=False).encode("utf-8"),
            file_name=f"providers_raw_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.csv",
            mime="text/csv",
        )

    # CSV Restore UI (Append-only, ID-checked)
    with st.expander("CSV Restore (Append-only, ID-checked)", expanded=False):
        st.caption(
            "WARNING: This tool only **appends** rows. "
            "Rows whose `id` already exists are **rejected**. No updates, no deletes."
        )
        uploaded = st.file_uploader("Upload CSV to append into `vendors`", type=["csv"], accept_multiple_files=False)

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            dry_run = st.checkbox("Dry run (validate only)", value=True)
        with col2:
            trim_strings = st.checkbox("Trim strings", value=True)
        with col3:
            normalize_phone = st.checkbox("Normalize phone to digits", value=True)
        with col4:
            auto_id = st.checkbox("Missing `id` ➜ autoincrement", value=True)

        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)
                with_id_df, without_id_df, rejected_ids, insertable_cols = _prepare_csv_for_append(
                    engine,
                    df_in,
                    normalize_phone=normalize_phone,
                    trim_strings=trim_strings,
                    treat_missing_id_as_autoincrement=auto_id,
                )

                planned_inserts = len(with_id_df) + len(without_id_df)

                st.write("**Validation summary**")
                st.write(
                    {
                        "csv_rows": int(len(df_in)),
                        "insertable_columns": insertable_cols,
                        "rows_with_explicit_id": int(len(with_id_df)),
                        "rows_autoincrement_id": int(len(without_id_df)),
                        "rows_rejected_due_to_existing_id": rejected_ids,
                        "planned_inserts": int(planned_inserts),
                    }
                )

                if dry_run:
                    st.success("Dry run complete. No changes applied.")
                else:
                    if planned_inserts == 0:
                        st.info("Nothing to insert (all rows rejected or CSV empty after filters).")
                    else:
                        inserted = _execute_append_only(engine, with_id_df, without_id_df, insertable_cols)
                        st.success(f"Inserted {inserted} row(s). Rejected existing id(s): {rejected_ids or 'None'}")
            except Exception as e:
                st.error(f"CSV restore failed: {e}")

    st.divider()
    st.subheader("Data cleanup")

    if st.button("Normalize phone numbers & Title Case (vendors + categories/services)"):
        def to_title(s: str | None) -> str:
            return ((s or "").strip()).title()

        TEXT_COLS_TO_TITLE = [
            "category",
            "service",
            "business_name",
            "contact_name",
            "address",
            "notes",
            "keywords",
        ]

        changed_vendors = 0
        try:
            with engine.begin() as conn:
                # --- vendors table ---
                rows = conn.execute(sql_text("SELECT * FROM vendors")).fetchall()
                for r in rows:
                    row = dict(r._mapping) if hasattr(r, "_mapping") else dict(r)
                    pid = int(row["id"])

                    vals = {c: to_title(row.get(c)) for c in TEXT_COLS_TO_TITLE}
                    vals["website"] = _sanitize_url((row.get("website") or "").strip())
                    vals["phone"] = _normalize_phone(row.get("phone") or "")
                    vals["id"] = pid

                    conn.execute(
                        sql_text(
                            """
                            UPDATE vendors
                               SET category=:category,
                                   service=NULLIF(:service,''),
                                   business_name=:business_name,
                                   contact_name=:contact_name,
                                   phone=:phone,
                                   address=:address,
                                   website=:website,
                                   notes=:notes,
                                   keywords=:keywords
                             WHERE id=:id
                            """
                        ),
                        vals,
                    )
                    changed_vendors += 1

                # --- categories table: retitle + reconcile duplicates by case ---
                cat_rows = conn.execute(sql_text("SELECT name FROM categories")).fetchall()
                for (old_name,) in cat_rows:
                    new_name = to_title(old_name)
                    if new_name != old_name:
                        conn.execute(sql_text("INSERT OR IGNORE INTO categories(name) VALUES(:n)"), {"n": new_name})
                        conn.execute(
                            sql_text("UPDATE vendors SET category=:new WHERE category=:old"),
                            {"new": new_name, "old": old_name},
                        )
                        conn.execute(sql_text("DELETE FROM categories WHERE name=:old"), {"old": old_name})

                # --- services table: retitle + reconcile duplicates by case ---
                svc_rows = conn.execute(sql_text("SELECT name FROM services")).fetchall()
                for (old_name,) in svc_rows:
                    new_name = to_title(old_name)
                    if new_name != old_name:
                        conn.execute(sql_text("INSERT OR IGNORE INTO services(name) VALUES(:n)"), {"n": new_name})
                        conn.execute(
                            sql_text("UPDATE vendors SET service=:new WHERE service=:old"),
                            {"new": new_name, "old": old_name},
                        )
                        conn.execute(sql_text("DELETE FROM services WHERE name=:old"), {"old": old_name})
            st.success(f"Vendors normalized: {changed_vendors}. Categories/services retitled and reconciled.")
        except Exception as e:
            st.error(f"Normalization failed: {e}")

    # Backfill timestamps (fix NULL and empty-string)
    if st.button("Backfill created_at/updated_at when missing"):
        try:
            now = datetime.utcnow().isoformat(timespec="seconds")
            with engine.begin() as conn:
                conn.execute(
                    sql_text(
                        """
                        UPDATE vendors
                           SET created_at = CASE WHEN created_at IS NULL OR created_at = '' THEN :now ELSE created_at END,
                               updated_at = CASE WHEN updated_at IS NULL OR updated_at = '' THEN :now ELSE updated_at END
                        """
                    ),
                    {"now": now},
                )
            st.success("Backfill complete.")
        except Exception as e:
            st.error(f"Backfill failed: {e}")

    # Trim extra whitespace across common text fields (preserves newlines in notes)
    if st.button("Trim whitespace in text fields (safe)"):
        try:
            changed = 0
            with engine.begin() as conn:
                rows = conn.execute(
                    sql_text(
                        """
                        SELECT id, category, service, business_name, contact_name, address, website, notes, keywords, phone
                        FROM vendors
                        """
                    )
                ).fetchall()

                def clean_soft(s: str | None) -> str:
                    s = (s or "").strip()
                    # collapse runs of spaces/tabs only; KEEP line breaks
                    s = re.sub(r"[ \t]+", " ", s)
                    return s

                for r in rows:
                    pid = int(r[0])
                    vals = {
                        "category": clean_soft(r[1]),
                        "service": clean_soft(r[2]),
                        "business_name": clean_soft(r[3]),
                        "contact_name": clean_soft(r[4]),
                        "address": clean_soft(r[5]),
                        "website": _sanitize_url(clean_soft(r[6])),
                        "notes": clean_soft(r[7]),  # preserves newlines
                        "keywords": clean_soft(r[8]),
                        "phone": r[9],  # leave phone unchanged here
                        "id": pid,
                    }
                    conn.execute(
                        sql_text(
                            """
                            UPDATE vendors
                               SET category=:category,
                                   service=NULLIF(:service,''),
                                   business_name=:business_name,
                                   contact_name=:contact_name,
                                   phone=:phone,
                                   address=:address,
                                   website=:website,
                                   notes=:notes,
                                   keywords=:keywords
                             WHERE id=:id
                            """
                        ),
                        vals,
                    )
                    changed += 1
            st.success(f"Whitespace trimmed on {changed} row(s).")
        except Exception as e:
            st.error(f"Trim failed: {e}")

# ---------- Debug
if _SHOW_DEBUG:
    with _tabs[-1]:
        st.subheader("Status & Secrets (debug)")
        st.json(engine_info)

        with engine.begin() as conn:
            vendors_cols = conn.execute(sql_text("PRAGMA table_info(vendors)")).fetchall()
            categories_cols = conn.execute(sql_text("PRAGMA table_info(categories)")).fetchall()
            services_cols = conn.execute(sql_text("PRAGMA table_info(services)")).fetchall()

            # --- Index presence (vendors) ---
            idx_rows = conn.execute(sql_text("PRAGMA index_list(vendors)")).fetchall()
            vendors_indexes = [
                {"seq": r[0], "name": r[1], "unique": bool(r[2]), "origin": r[3], "partial": bool(r[4])} for r in idx_rows
            ]

            # --- Null timestamp counts (quick sanity) ---
            created_at_nulls = conn.execute(
                sql_text("SELECT COUNT(*) FROM vendors WHERE created_at IS NULL OR created_at=''")
            ).scalar() or 0
            updated_at_nulls = conn.execute(
                sql_text("SELECT COUNT(*) FROM vendors WHERE updated_at IS NULL OR updated_at=''")
            ).scalar() or 0

            counts = {
                "vendors": conn.execute(sql_text("SELECT COUNT(*) FROM vendors")).scalar() or 0,
                "categories": conn.execute(sql_text("SELECT COUNT(*) FROM categories")).scalar() or 0,
                "services": conn.execute(sql_text("SELECT COUNT(*) FROM services")).scalar() or 0,
            }

        st.subheader("DB Probe")
        st.json(
            {
                "vendors_columns": [c[1] for c in vendors_cols],
                "categories_columns": [c[1] for c in categories_cols],
                "services_columns": [c[1] for c in services_cols],
                "counts": counts,
                "vendors_indexes": vendors_indexes,
                "timestamp_nulls": {"created_at": int(created_at_nulls), "updated_at": int(updated_at_nulls)},
            }
        )

        # --- Null timestamp counts (quick sanity) ---
        created_at_nulls = conn.execute(
            sql_text("SELECT COUNT(*) FROM vendors WHERE created_at IS NULL OR created_at=''")
        ).scalar() or 0
        updated_at_nulls = conn.execute(
            sql_text("SELECT COUNT(*) FROM vendors WHERE updated_at IS NULL OR updated_at=''")
        ).scalar() or 0

        counts = {
            "vendors": conn.execute(sql_text("SELECT COUNT(*) FROM vendors")).scalar() or 0,
            "categories": conn.execute(sql_text("SELECT COUNT(*) FROM categories")).scalar() or 0,
            "services": conn.execute(sql_text("SELECT COUNT(*) FROM services")).scalar() or 0,
        }

    st.subheader("DB Probe")
    st.json(
        {
            "vendors_columns": [c[1] for c in vendors_cols],
            "categories_columns": [c[1] for c in categories_cols],
            "services_columns": [c[1] for c in services_cols],
            "counts": counts,
            "vendors_indexes": vendors_indexes,
            "timestamp_nulls": {"created_at": int(created_at_nulls), "updated_at": int(updated_at_nulls)},
        }
    )

