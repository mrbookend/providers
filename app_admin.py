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

# -------- Early Boot (dialect/version, secrets, engine builder, diagnostics) --------

# libsql dialect version (ok if missing)
try:
    import sqlalchemy_libsql as _lib
    _lib_ver = getattr(_lib, "__version__", "unknown")
except Exception:
    _lib = None
    _lib_ver = "n/a"

# ==== BEGIN: Strategy + required secrets (validated) ====
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

# Hard-require secrets only when we intend to sync
if _DB_STRATEGY in ("embedded_replica", "replica", "sync"):
    try:
        _validate_sync_url(_TURSO_URL)
    except Exception as _e:
        st.error(f"Bad TURSO_DATABASE_URL: {type(_e).__name__}: {_e}")
        st.stop()
    if not _TURSO_TOKEN:
        st.error("Missing required secret: TURSO_AUTH_TOKEN")
        st.stop()
# ==== END: Strategy + required secrets (validated) ====


# Helper: mask long strings for display
def _mask(u: str | None, keep: int = 16) -> str:
    if not u:
        return ""
    return u[:keep] + "…" if len(u) > keep else u

# ==== BEGIN: Version Banner (enhanced) ====
import importlib
import importlib.metadata as _im

_lib_ver_display = _lib_ver
_lib_file = "n/a"
try:
    # Prefer the package version as installed in the environment
    _lib_ver_pkg = _im.version("sqlalchemy-libsql")
    _lib_ver_display = _lib_ver_pkg or _lib_ver_display
except Exception:
    pass

try:
    _lib_mod = importlib.import_module("sqlalchemy_libsql")
    _lib_file = getattr(_lib_mod, "__file__", "n/a")
except Exception:
    pass

if bool(st.secrets.get("SHOW_STATUS", True)):
    try:
        st.sidebar.info(
            "Versions | "
            f"py {sys.version.split()[0]} | "
            f"streamlit {st.__version__} | "
            f"pandas {pd.__version__} | "
            f"SA {sqlalchemy.__version__} | "
            f"libsql {(_lib_ver_display or 'n/a')} | "
            f"requests {requests.__version__}"
        )
        with st.sidebar.expander("libsql details"):
            st.write({"module_file": _lib_file, "pkg_version": _lib_ver_display})
    except Exception as _e:
        st.sidebar.warning(f"Version banner failed: {_e}")
# ==== END: Version Banner (enhanced) ====

# ==== BEGIN: Engine builder (embedded replica w/ libsql sync) ====
def build_engine_and_probe() -> tuple[Engine, dict]:
    # Base engine always opens the local file using libsql dialect
    engine_url = f"sqlite+libsql:///{_EMBEDDED_PATH}"
    dbg = {
        "host": platform.node() or "localhost",
        "strategy": _DB_STRATEGY,
        "python": sys.version.split()[0],
        "sqlalchemy_url": engine_url,
        "embedded_path": _EMBEDDED_PATH,
        "libsql_ver": (_lib_ver if isinstance(_lib_ver, str) else "unknown"),
        "sync_url_scheme": "",
    }

    connect_args: Dict[str, Any] = {}

    if _DB_STRATEGY in ("embedded_replica", "replica", "sync"):
        valid_sync = _validate_sync_url(_TURSO_URL)
        dbg["sync_url_scheme"] = valid_sync.split("://", 1)[0] + "://"
        connect_args["sync_url"] = valid_sync
        connect_args["auth_token"] = _TURSO_TOKEN
    elif _DB_STRATEGY == "remote_only":
        # Direct remote, no embedded file
        engine_url = _validate_sync_url(_TURSO_URL)
        dbg["sqlalchemy_url"] = engine_url
        dbg["embedded_path"] = ""
        dbg["sync_url_scheme"] = "libsql://"
        connect_args["auth_token"] = _TURSO_TOKEN
    else:
        # "embedded_only": no sync
        dbg["sync_url_scheme"] = ""

    eng = create_engine(engine_url, connect_args=connect_args)

    # Quick sanity check (will raise early if driver/url is bad)
    with eng.connect() as cx:
        cx.exec_driver_sql("PRAGMA journal_mode")

    return eng, dbg
# ==== END: Engine builder (embedded replica w/ libsql sync) ====


# ==== BEGIN: SINGLE canonical engine builder (drop-in) ====
def build_engine_and_probe() -> tuple[Engine, Dict]:
    """
    Canonical engine builder used across the entire app.
    - Strategy 'embedded_replica' requires TURSO_DATABASE_URL and TURSO_AUTH_TOKEN
    - Otherwise falls back to plain sqlite using the same embedded path
    """
    url_remote = st.secrets.get("TURSO_DATABASE_URL")
    token      = st.secrets.get("TURSO_AUTH_TOKEN")
    # Default path if not provided (works locally and on Streamlit Cloud)
    embedded   = st.secrets.get("EMBEDDED_DB_PATH", "/mount/src/providers/vendors-embedded.db")
    strategy   = _strategy

    # Force absolute path on Cloud (defensive)
    if not embedded.startswith("/"):
        embedded = "/mount/src/providers/" + embedded.lstrip("/")

    # Ensure embedded DB dir exists (avoid 'unable to open database file')
    try:
        os.makedirs(os.path.dirname(embedded), exist_ok=True)
    except Exception as _mkerr:
        st.warning(f"Could not ensure embedded DB dir exists: {_mkerr.__class__.__name__}: {_mkerr}")

    use_replica = (strategy == "embedded_replica") and bool(url_remote and token)

    if use_replica:
        if _lib is None:
            st.error(
                "sqlalchemy_libsql module is not available, but strategy requires it "
                "(embedded_replica). Check requirements and reinstall."
            )
            st.stop()
        sqlalchemy_url = f"sqlite+libsql:///{embedded}"
        # normalize sync_url (drop query params like ?secure=true)
        sync_url = (url_remote or "").split("?", 1)[0]
        connect_args = {"sync_url": sync_url, "auth_token": token}
    else:
        sqlalchemy_url = f"sqlite:///{embedded}"
        connect_args = {}

    # ==== BEGIN: dbg dict + enhanced libsql version patch (after if/else, before engine connect) ====
    dbg = {
        "host": platform.node(),
        "strategy": strategy,
        "python": sys.version.split()[0],
        "sqlalchemy_url": sqlalchemy_url,
        "embedded_path": embedded,
        "sync_url": _mask(url_remote),
        "libsql_ver": _lib_ver,  # may be stale inside module; patch below fixes it
    }

    # Patch dbg with the actual installed package version and import path
    import importlib
    import importlib.metadata as _im
    try:
        _lib_ver_pkg = _im.version("sqlalchemy-libsql")
    except Exception:
        _lib_ver_pkg = None
    try:
        _lib_mod = importlib.import_module("sqlalchemy_libsql")
        _lib_file = getattr(_lib_mod, "__file__", "n/a")
    except Exception:
        _lib_file = "n/a"

    dbg["libsql_ver"] = _lib_ver_pkg or dbg.get("libsql_ver", "n/a")
    dbg["libsql_module_file"] = _lib_file
    # ==== END: dbg dict + enhanced libsql version patch ====
    # ==== BEGIN: engine connect + return (append after dbg patch) ====
    try:
        eng = create_engine(sqlalchemy_url, connect_args=connect_args)
        last_err = None
        for attempt in range(5):
            try:
                with eng.connect() as cx:
                    cx.execute(sql_text("SELECT 1"))
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.6 * (attempt + 1))
        if last_err:
            raise last_err
    except Exception as e:
        st.error(f"DB init failed: {e.__class__.__name__}: {e}")
        with st.expander("Diagnostics"):
            st.json(dbg)
        st.stop()

    return eng, dbg
# ==== END: SINGLE canonical engine builder (drop-in, final) ====
    # Create engine and prove connectivity with simple backoff (Cloud can be slow to boot)
    try:
        eng = create_engine(sqlalchemy_url, connect_args=connect_args)
        last_err = None
        for attempt in range(5):
            try:
                with eng.connect() as cx:
                    cx.execute(sql_text("SELECT 1"))
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.6 * (attempt + 1))
        if last_err:
            raise last_err
    except Exception as e:
        st.error(f"DB init failed: {e.__class__.__name__}: {e}")
        with st.expander("Diagnostics"):
            st.json(dbg)
        st.stop()

    return eng, dbg
# ==== END: SINGLE canonical engine builder (drop-in) ====

    # Create engine and prove connectivity with simple backoff (Cloud can be slow to boot)
    try:
        eng = create_engine(sqlalchemy_url, connect_args=connect_args)
        last_err = None
        for attempt in range(5):
            try:
                with eng.connect() as cx:
                    cx.execute(sql_text("SELECT 1"))
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.6 * (attempt + 1))
        if last_err:
            raise last_err
    except Exception as e:
        st.error(f"DB init failed: {e.__class__.__name__}: {e}")
        with st.expander("Diagnostics"):
            st.json(dbg)
        st.stop()

    return eng, dbg

# ==== BEGIN: Engine init + diagnostics (call once) ====
try:
    ENGINE, _DB_DBG = build_engine_and_probe()
    st.sidebar.success("DB ready")

    # Stash for reuse
    st.session_state["ENGINE"] = ENGINE
    st.session_state["DB_DBG"] = _DB_DBG

    with st.expander("Boot diagnostics (ENGINE + secrets)"):
        st.json(_DB_DBG)

    st.success("App reached post-boot marker ✅")  # proves we got past engine init

    # Optional: quick vendors count (gate with SHOW_COUNT; table may not exist yet)
    if bool(st.secrets.get("SHOW_COUNT", True)):
        try:
            with ENGINE.connect() as cx:
                cnt = cx.exec_driver_sql("SELECT COUNT(*) FROM vendors").scalar()
            st.info(f"Vendors table row count: {cnt}")
        except Exception as _e:
            st.warning(f"Quick vendors count failed: {type(_e).__name__}: {_e}")

except Exception as e:
    st.error(f"Database init failed: {e.__class__.__name__}: {e}")
    st.stop()
# ==== END: Engine init + diagnostics (call once) ====

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
# DB helpers (schema + IO)
# -----------------------------
REQUIRED_VENDOR_COLUMNS: List[str] = ["business_name", "category"]  # service optional

def ensure_schema(engine: Engine) -> None:
    stmts = [
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
        """
        CREATE TABLE IF NOT EXISTS categories (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT UNIQUE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS services (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT UNIQUE
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_vendors_cat ON vendors(category)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_bus ON vendors(business_name)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_kw  ON vendors(keywords)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_bus_lower ON vendors(lower(business_name))",
        "CREATE INDEX IF NOT EXISTS idx_vendors_cat_lower ON vendors(lower(category))",
        "CREATE INDEX IF NOT EXISTS idx_vendors_svc_lower ON vendors(lower(service))",
        "CREATE INDEX IF NOT EXISTS idx_vendors_phone ON vendors(phone)",
    ]
    with engine.begin() as conn:
        for s in stmts:
            conn.execute(sql_text(s))


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

    # Display-friendly phone; storage remains digits
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


# -----------------------------
# CSV Restore helpers (append-only, ID-checked)
# -----------------------------
def _get_table_columns(engine: Engine, table: str) -> list[str]:
    with engine.connect() as conn:
        res = conn.execute(sql_text(f"SELECT * FROM {table} LIMIT 0"))
        return list(res.keys())


def _fetch_existing_ids(engine: Engine, table: str = "vendors") -> set[int]:
    with engine.connect() as conn:
        rows = conn.execute(sql_text(f"SELECT id FROM {table}")).all()
    return {int(r[0]) for r in rows if r[0] is not None}


def _prepare_csv_for_append(
    engine: Engine,
    csv_df: pd.DataFrame,
    *,
    normalize_phone: bool,
    trim_strings: bool,
    treat_missing_id_as_autoincrement: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[str]]:
    """
    Returns: (with_id_df, without_id_df, rejected_existing_ids, insertable_columns)
    DataFrames are already filtered to allowed columns and safe to insert.
    """
    df = csv_df.copy()

    # Trim strings
    if trim_strings:
        for c in df.columns:
            if pd.api.types.is_object_dtype(df[c]):
                df[c] = df[c].astype(str).str.strip()

    # Normalize phone to digits
    if normalize_phone and "phone" in df.columns:
        df["phone"] = df["phone"].astype(str).str.replace(r"\D+", "", regex=True)

    db_cols = _get_table_columns(engine, "vendors")
    insertable_cols = [c for c in df.columns if c in db_cols]

    # Required columns present?
    missing_req = [c for c in REQUIRED_VENDOR_COLUMNS if c not in df.columns]
    if missing_req:
        raise ValueError(f"Missing required column(s) in CSV: {missing_req}")

    # Handle id column
    has_id = "id" in df.columns
    existing_ids = _fetch_existing_ids(engine)

    if has_id:
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
        # Reject rows colliding with existing ids
        mask_conflict = df["id"].notna() & df["id"].astype("Int64").astype("int", errors="ignore").isin(existing_ids)
        rejected_existing_ids = df.loc[mask_conflict, "id"].dropna().astype(int).tolist()
        df_ok = df.loc[~mask_conflict].copy()

        # Split by having id vs. not
        with_id_df = df_ok[df_ok["id"].notna()].copy()
        without_id_df = df_ok[df_ok["id"].isna()].copy() if treat_missing_id_as_autoincrement else pd.DataFrame(columns=df.columns)
    else:
        rejected_existing_ids = []
        with_id_df = pd.DataFrame(columns=df.columns)
        without_id_df = df.copy()

    # Limit to insertable columns and coerce NaN->None for DB
    def _prep_cols(d: pd.DataFrame, drop_id: bool) -> pd.DataFrame:
        cols = [c for c in insertable_cols if (c != "id" if drop_id else True)]
        if not cols:
            return pd.DataFrame(columns=[])
        dd = d[cols].copy()
        for c in cols:
            dd[c] = dd[c].where(pd.notnull(dd[c]), None)
        return dd

    with_id_df = _prep_cols(with_id_df, drop_id=False)
    without_id_df = _prep_cols(without_id_df, drop_id=True)

    # Duplicate ids inside the CSV itself?
    if "id" in csv_df.columns:
        dup_ids = (
            csv_df["id"]
            .pipe(pd.to_numeric, errors="coerce")
            .dropna()
            .astype(int)
            .duplicated(keep=False)
        )
        if dup_ids.any():
            dups = sorted(csv_df.loc[dup_ids, "id"].dropna().astype(int).unique().tolist())
            raise ValueError(f"Duplicate id(s) inside CSV: {dups}")

    return with_id_df, without_id_df, rejected_existing_ids, insertable_cols


def _execute_append_only(
    engine: Engine,
    with_id_df: pd.DataFrame,
    without_id_df: pd.DataFrame,
    insertable_cols: list[str],
) -> int:
    """Executes INSERTs in a single transaction. Returns total inserted rows."""
    inserted = 0
    with engine.begin() as conn:
        # with explicit id
        if not with_id_df.empty:
            cols = list(with_id_df.columns)  # includes 'id' by construction
            placeholders = ", ".join(":" + c for c in cols)
            stmt = sql_text(f"INSERT INTO vendors ({', '.join(cols)}) VALUES ({placeholders})")
            conn.execute(stmt, with_id_df.to_dict(orient="records"))
            inserted += len(with_id_df)

        # without id (autoincrement)
        if not without_id_df.empty:
            cols = list(without_id_df.columns)  # 'id' removed already
            placeholders = ", ".join(":" + c for c in cols)
            stmt = sql_text(f"INSERT INTO vendors ({', '.join(cols)}) VALUES ({placeholders})")
            conn.execute(stmt, without_id_df.to_dict(orient="records"))
            inserted += len(without_id_df)

    return inserted


# -----------------------------
# SINGLE engine wiring for rest of app
# -----------------------------
engine: Engine = ENGINE
engine_info: Dict = _DB_DBG

# Ensure schema on the single engine
ensure_schema(engine)

# Apply WAL PRAGMAs for local SQLite (NOT for libsql driver)
try:
    if engine.dialect.name == "sqlite" and getattr(engine.dialect, "driver", "") != "libsql":
        with engine.begin() as _conn:
            _conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
            _conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
except Exception:
    # Best-effort; ignore if unavailable
    pass


# -----------------------------
# UI
# -----------------------------
_tabs = st.tabs(
    [
        "Browse Vendors",
        "Add / Edit / Delete Vendor",
        "Category Admin",
        "Service Admin",
        "Maintenance",
        "Debug",
    ]
)

# ---------- Browse
with _tabs[0]:
    df = load_df(engine)

    # --- Build a lowercase search blob once (guarded) ---
    if "_blob" not in df.columns:
        parts = [
            df.get(c, pd.Series("", index=df.index)).astype(str)
            for c in ["business_name", "category", "service", "contact_name", "phone", "address", "website", "notes", "keywords"]
        ]
        df["_blob"] = pd.concat(parts, axis=1).agg(" ".join, axis=1).str.lower()

    # --- Search input at 25% width (table remains full width) ---
    left, right = st.columns([1, 3])  # 25% / 75% split for this row only
    with left:
        q = st.text_input(
            "Search",
            placeholder="Search providers… (press Enter)",
            label_visibility="collapsed",
            key="q",
        )

    # Fast local filter using the prebuilt blob (no regex)
    qq = (st.session_state.get("q") or "").strip().lower()
    if qq:
        filtered = df[df["_blob"].str.contains(qq, regex=False, na=False)]
    else:
        filtered = df

    view_cols = [
        "id",
        "category",
        "service",
        "business_name",
        "contact_name",
        "phone_fmt",
        "address",
        "website",
        "notes",
        "keywords",
    ]

    vdf = filtered[view_cols].rename(columns={"phone_fmt": "phone"})

    # Read-only table with clickable website links
    st.dataframe(
        vdf,
        use_container_width=True,
        hide_index=True,
        column_config={
            "business_name": st.column_config.TextColumn("Provider"),
            "website": st.column_config.LinkColumn("website"),
            "notes": st.column_config.TextColumn(width=420),
            "keywords": st.column_config.TextColumn(width=300),
        },
    )

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    st.download_button(
        "Download filtered view (CSV)",
        data=vdf.to_csv(index=False).encode("utf-8"),
        file_name=f"providers_{ts}.csv",
        mime="text/csv",
    )

# ---------- Add/Edit/Delete Vendor
with _tabs[1]:
    # ===== Add Vendor =====
    st.subheader("Add Vendor")
    _init_add_form_defaults()
    _apply_add_reset_if_needed()  # apply queued reset BEFORE creating widgets

    cats = list_names(engine, "categories")
    servs = list_names(engine, "services")

    add_form_key = f"add_vendor_form_{st.session_state['add_form_version']}"
    with st.form(add_form_key, clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Provider *", key="add_business_name")

            # Category select—options include "" placeholder; do NOT pass index when using session_state defaults
            _add_cat_options = [""] + (cats or [])
            if (st.session_state.get("add_category") or "") not in _add_cat_options:
                st.session_state["add_category"] = ""
            st.selectbox("Category *", options=_add_cat_options, key="add_category", placeholder="Select category")

            # Service select—same pattern
            _add_svc_options = [""] + (servs or [])
            if (st.session_state.get("add_service") or "") not in _add_svc_options:
                st.session_state["add_service"] = ""
            st.selectbox("Service (optional)", options=_add_svc_options, key="add_service")

            st.text_input("Contact Name", key="add_contact_name")
            st.text_input("Phone (10 digits or blank)", key="add_phone")
        with col2:
            st.text_area("Address", height=80, key="add_address")
            st.text_input("Website (https://…)", key="add_website")
            st.text_area("Notes", height=100, key="add_notes")
            st.text_input("Keywords (comma separated)", key="add_keywords")

        submitted = st.form_submit_button("Add Vendor")

    if submitted:
        add_nonce = _nonce("add")
        if st.session_state.get("add_last_done") == add_nonce:
            st.info("Add already processed.")
            st.stop()

        business_name = (st.session_state["add_business_name"] or "").strip()
        category      = (st.session_state["add_category"] or "").strip()
        service       = (st.session_state["add_service"] or "").strip()
        contact_name  = (st.session_state["add_contact_name"] or "").strip()
        phone_norm    = _normalize_phone(st.session_state["add_phone"])
        address       = (st.session_state["add_address"] or "").strip()
        website       = _sanitize_url(st.session_state["add_website"])
        notes         = (st.session_state["add_notes"] or "").strip()
        keywords      = (st.session_state["add_keywords"] or "").strip()

        # Minimal-change validation: phone must be 10 digits or blank
        if phone_norm and len(phone_norm) != 10:
            st.error("Phone must be 10 digits or blank.")
        elif not business_name or not category:
            st.error("Business Name and Category are required.")
        else:
            try:
                now = datetime.utcnow().isoformat(timespec="seconds")
                _exec_with_retry(
                    engine,
                    """
                    INSERT INTO vendors(category, service, business_name, contact_name, phone, address,
                                        website, notes, keywords, created_at, updated_at, updated_by)
                    VALUES(:category, NULLIF(:service, ''), :business_name, :contact_name, :phone, :address,
                           :website, :notes, :keywords, :now, :now, :user)
                    """,
                    {
                        "category": category,
                        "service": service,
                        "business_name": business_name,
                        "contact_name": contact_name,
                        "phone": phone_norm,
                        "address": address,
                        "website": website,
                        "notes": notes,
                        "keywords": keywords,
                        "now": now,
                        "user": os.getenv("USER", "admin"),
                    },
                )
                st.session_state["add_last_done"] = add_nonce
                st.success(f"Vendor added: {business_name}")
                _queue_add_form_reset()
                _nonce_rotate("add")
                st.rerun()
            except Exception as e:
                st.error(f"Add failed: {e}")

    st.divider()
    st.subheader("Edit / Delete Vendor")

    df_all = load_df(engine)

    if df_all.empty:
        st.info("No vendors yet. Use 'Add Vendor' above to create your first record.")
    else:
        # Init + apply resets BEFORE rendering widgets
        _init_edit_form_defaults()
        _init_delete_form_defaults()
        _apply_edit_reset_if_needed()
        _apply_delete_reset_if_needed()

        # ----- EDIT: ID-backed selection with format_func -----
        ids = df_all["id"].astype(int).tolist()
        id_to_row = {int(r["id"]): r for _, r in df_all.iterrows()}

        def _fmt_vendor(i: int | None) -> str:
            if i is None:
                return "— Select —"
            r = id_to_row.get(int(i), None)
            if r is None:
                return f"{i}"
            cat = (r.get("category") or "")
            svc = (r.get("service") or "")
            tail = " / ".join([x for x in (cat, svc) if x]).strip(" /")
            name = str(r.get("business_name") or "")
            return f"{name} — {tail}" if tail else name

        st.selectbox(
            "Select provider to edit (type to search)",
            options=[None] + ids,
            format_func=_fmt_vendor,
            key="edit_vendor_id",
        )

        # Prefill only when selection changes
        if st.session_state["edit_vendor_id"] is not None:
            if st.session_state["edit_last_loaded_id"] != st.session_state["edit_vendor_id"]:
                row = id_to_row[int(st.session_state["edit_vendor_id"])]
                st.session_state.update({
                    "edit_business_name": row.get("business_name") or "",
                    "edit_category": row.get("category") or "",
                    "edit_service": row.get("service") or "",
                    "edit_contact_name": row.get("contact_name") or "",
                    "edit_phone": row.get("phone") or "",
                    "edit_address": row.get("address") or "",
                    "edit_website": row.get("website") or "",
                    "edit_notes": row.get("notes") or "",
                    "edit_keywords": row.get("keywords") or "",
                    "edit_row_updated_at": row.get("updated_at") or "",
                    "edit_last_loaded_id": st.session_state["edit_vendor_id"],
                })

        # -------- Edit form --------
        edit_form_key = f"edit_vendor_form_{st.session_state['edit_form_version']}"
        with st.form(edit_form_key, clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Provider *", key="edit_business_name")

                cats = list_names(engine, "categories")
                servs = list_names(engine, "services")

                _edit_cat_options = [""] + (cats or [])
                if (st.session_state.get("edit_category") or "") not in _edit_cat_options:
                    st.session_state["edit_category"] = ""
                st.selectbox("Category *", options=_edit_cat_options, key="edit_category", placeholder="Select category")

                _edit_svc_options = [""] + (servs or [])
                if (st.session_state.get("edit_service") or "") not in _edit_svc_options:
                    st.session_state["edit_service"] = ""
                st.selectbox("Service (optional)", options=_edit_svc_options, key="edit_service")

                st.text_input("Contact Name", key="edit_contact_name")
                st.text_input("Phone (10 digits or blank)", key="edit_phone")
            with col2:
                st.text_area("Address", height=80, key="edit_address")
                st.text_input("Website (https://…)", key="edit_website")
                st.text_area("Notes", height=100, key="edit_notes")
                st.text_input("Keywords (comma separated)", key="edit_keywords")

            edited = st.form_submit_button("Save Changes")

        if edited:
            edit_nonce = _nonce("edit")
            if st.session_state.get("edit_last_done") == edit_nonce:
                st.info("Edit already processed.")
                st.stop()

            vid = st.session_state.get("edit_vendor_id")
            if vid is None:
                st.error("Select a vendor first.")
            else:
                bn  = (st.session_state["edit_business_name"] or "").strip()
                cat = (st.session_state["edit_category"] or "").strip()
                phone_norm = _normalize_phone(st.session_state["edit_phone"])
                if phone_norm and len(phone_norm) != 10:
                    st.error("Phone must be 10 digits or blank.")
                elif not bn or not cat:
                    st.error("Business Name and Category are required.")
                else:
                    try:
                        prev_updated = st.session_state.get("edit_row_updated_at") or ""
                        now = datetime.utcnow().isoformat(timespec="seconds")
                        res = _exec_with_retry(engine, """
                            UPDATE vendors
                               SET category=:category,
                                   service=NULLIF(:service, ''),
                                   business_name=:business_name,
                                   contact_name=:contact_name,
                                   phone=:phone,
                                   address=:address,
                                   website=:website,
                                   notes=:notes,
                                   keywords=:keywords,
                                   updated_at=:now,
                                   updated_by=:user
                             WHERE id=:id AND (updated_at=:prev_updated OR :prev_updated='')
                        """, {
                            "category": cat,
                            "service": (st.session_state["edit_service"] or "").strip(),
                            "business_name": bn,
                            "contact_name": (st.session_state["edit_contact_name"] or "").strip(),
                            "phone": phone_norm,
                            "address": (st.session_state["edit_address"] or "").strip(),
                            "website": _sanitize_url(st.session_state["edit_website"]),
                            "notes": (st.session_state["edit_notes"] or "").strip(),
                            "keywords": (st.session_state["edit_keywords"] or "").strip(),
                            "now": now, "user": os.getenv("USER", "admin"),
                            "id": int(vid),
                            "prev_updated": prev_updated,
                        })
                        rowcount = res.rowcount or 0

                        if rowcount == 0:
                            st.warning("No changes applied (stale selection or already updated). Refresh and try again.")
                        else:
                            st.session_state["edit_last_done"] = edit_nonce
                            st.success(f"Vendor updated: {bn}")
                            _queue_edit_form_reset()
                            _nonce_rotate("edit")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Update failed: {e}")

        st.markdown("---")
        # Use separate delete selection (ID-backed similar approach could be added later)
        sel_label_del = st.selectbox(
            "Select provider to delete (type to search)",
            options=["— Select —"] + [ _fmt_vendor(i) for i in ids ],
            key="delete_provider_label",
        )
        if sel_label_del != "— Select —":
            rev = { _fmt_vendor(i): i for i in ids }
            st.session_state["delete_vendor_id"] = int(rev.get(sel_label_del))
        else:
            st.session_state["delete_vendor_id"] = None

        del_form_key = f"delete_vendor_form_{st.session_state['delete_form_version']}"
        with st.form(del_form_key, clear_on_submit=False):
            deleted = st.form_submit_button("Delete Vendor")

        if deleted:
            del_nonce = _nonce("delete")
            if st.session_state.get("delete_last_done") == del_nonce:
                st.info("Delete already processed.")
                st.stop()

            vid = st.session_state.get("delete_vendor_id")
            if vid is None:
                st.error("Select a vendor first.")
            else:
                try:
                    row = df_all.loc[df_all["id"] == int(vid)]
                    prev_updated = (row.iloc[0]["updated_at"] if not row.empty else "") or ""
                    res = _exec_with_retry(engine, """
                        DELETE FROM vendors
                         WHERE id=:id AND (updated_at=:prev_updated OR :prev_updated='')
                    """, {"id": int(vid), "prev_updated": prev_updated})
                    rowcount = res.rowcount or 0

                    if rowcount == 0:
                        st.warning("No delete performed (stale selection). Refresh and try again.")
                    else:
                        st.session_state["delete_last_done"] = del_nonce
                        st.success("Vendor deleted.")
                        _queue_delete_form_reset()
                        _nonce_rotate("delete")
                        st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")

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
                        _exec_with_retry(engine, "UPDATE categories SET name=:new WHERE name=:old", {"new": new.strip(), "old": old})
                        _exec_with_retry(engine, "UPDATE vendors SET category=:new WHERE category=:old", {"new": new.strip(), "old": old})
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
                    repl = st.selectbox("Reassign vendors to…", options=repl_options, key="cat_reassign_to")  # no index
                    if st.button("Reassign vendors then delete", key="cat_reassign_btn"):
                        if repl == "— Select —":
                            st.error("Choose a category to reassign to.")
                        else:
                            try:
                                _exec_with_retry(engine, "UPDATE vendors SET category=:r WHERE category=:t", {"r": repl, "t": tgt})
                                _exec_with_retry(engine, "DELETE FROM categories WHERE name=:t", {"t": tgt})
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
                        _exec_with_retry(engine, "UPDATE services SET name=:new WHERE name=:old", {"new": new.strip(), "old": old})
                        _exec_with_retry(engine, "UPDATE vendors SET service=:new WHERE service=:old", {"new": new.strip(), "old": old})
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
                    repl = st.selectbox("Reassign vendors to…", options=repl_options, key="svc_reassign_to")  # no index
                    if st.button("Reassign vendors then delete service", key="svc_reassign_btn"):
                        if repl == "— Select —":
                            st.error("Choose a service to reassign to.")
                        else:
                            try:
                                _exec_with_retry(engine, "UPDATE vendors SET service=:r WHERE service=:t", {"r": repl, "t": tgt})
                                _exec_with_retry(engine, "DELETE FROM services WHERE name=:t", {"t": tgt})
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
with _tabs[5]:
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
