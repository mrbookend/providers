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

import math
import json
import pandas as pd
import requests
import sqlalchemy
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# ---- Streamlit page config MUST be first Streamlit call ----
st.set_page_config(page_title="Providers Admin", layout="wide", initial_sidebar_state="expanded")

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

# --- Secrets / strategy (validated) ---
_DB_STRATEGY   = str(st.secrets.get("DB_STRATEGY", "embedded_replica")).strip().lower()
_TURSO_URL     = str(st.secrets.get("TURSO_DATABASE_URL", "")).strip()
_TURSO_TOKEN   = str(st.secrets.get("TURSO_AUTH_TOKEN", "")).strip()
_EMBEDDED_PATH = str(st.secrets.get("EMBEDDED_DB_PATH", "/mount/src/providers/vendors-embedded.db")).strip()
# --- URL validation for Turso sync (must be libsql://<host>[...]) ---
from urllib.parse import urlparse

def _validate_sync_url(u: str) -> str:
    """
    Validate and normalize TURSO_DATABASE_URL for sync:
      - non-empty
      - scheme must be exactly libsql (case-insensitive)
      - must NOT start with sqlite+libsql:// (that's an engine URL, not a sync URL)
      - hostname must be present
    Returns the stripped string value (original casing for host preserved).
    """
    if not u or not str(u).strip():
        raise ValueError("TURSO_DATABASE_URL is empty; expected libsql://<host>")

    s = str(u).strip().strip('"').strip("'")  # trim whitespace and accidental quotes

    # Reject engine-style scheme even if cased weirdly
    if s.lower().startswith("sqlite+libsql://"):
        raise ValueError("TURSO_DATABASE_URL must start with libsql:// (not sqlite+libsql://)")

    p = urlparse(s)
    scheme = (p.scheme or "").lower()
    if scheme != "libsql":
        raise ValueError(f"Unsupported sync URL scheme: {p.scheme or '(missing)'}:// (expected libsql://)")

    # Require a hostname; tolerate path/query/port (Turso allows them)
    if not (p.hostname and p.hostname.strip()):
        raise ValueError("TURSO_DATABASE_URL is missing a hostname")

    return s  # return normalized string, not the parsed object

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

# --- URL masker: never leak tokens/paths; show scheme://host[:port] only ---
def _mask_sync_url(u: str) -> str:
    if not u:
        return ""
    try:
        p = urlparse(u)
        host = p.hostname or ""
        port = f":{p.port}" if p.port else ""
        scheme = (p.scheme or "").lower()
        return f"{scheme}://{host}{port}"
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
        # Sidebar runtime overrides (session-only toggle; takes effect next rerun)
        with st.sidebar.expander("Admin runtime toggles", expanded=False):
            st.checkbox("Show debug (this session only)", key="ADMIN_RUNTIME_DEBUG")
        if st.session_state.get("ADMIN_RUNTIME_DEBUG"):
            _SHOW_DEBUG = True  # session override (safe; defaults still from secrets)

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

        # Stash for reuse (UI context only)
        st.session_state["ENGINE"] = ENGINE
        st.session_state["DB_DBG"]  = _DB_DBG
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

# ==== BEGIN: CKW Config & Cache (INSERT after imports/secrets) ====
from functools import lru_cache

# Version tag for audit and drift control; override via secrets if set
CKW_VERSION = str(st.secrets.get("CKW_VERSION", "2025-10-15")).strip() or "2025-10-15"
CKW_MAX_TERMS = int(st.secrets.get("CKW_MAX_TERMS", 40))

@st.cache_data(show_spinner=False)
def _load_ckw_rules() -> dict:
    """
    Loads curated rules from secrets (JSON string or TOML tables).
    Expected shape (examples):
      {
        "_global": ["verified","professional"],
        "Plumber": ["emergency","leak"],
        "Electrician": ["licensed","panel"],
        "General": { "Handyman": ["repair","home","fix"] }
      }
    """
    raw = st.secrets.get("CKW_RULES", "")
    try:
        if isinstance(raw, str) and raw.strip().startswith("{"):
            return json.loads(raw)
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return {"_global": []}
# ==== END: CKW Config & Cache ====

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

def _get_secret(name: str, default: str | None = None) -> str | dict | None:
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

def _resolve_json(name: str, code_default: dict | None = None) -> dict:
    raw = _get_secret(name, None)
    if raw is None:
        return code_default or {}
    if isinstance(raw, dict):
        return raw
    # try JSON then TOML (py3.11 tomllib)
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        import tomllib as _toml  # py311+
    except Exception:
        _toml = None
    if _toml:
        try:
            return _toml.loads(raw)
        except Exception:
            pass
    return code_default or {}

def _ct_equals(a: str, b: str) -> bool:
    """Constant-time string compare for secrets."""
    return hmac.compare_digest((a or ""), (b or ""))

def _updated_by() -> str:
    """Single source of truth for updated_by stamps."""
    try:
        v = str(st.secrets.get("ADMIN_USER", "")).strip()
    except Exception:
        v = ""
    if v:
        return v
    return os.getenv("USER", "admin")

# -----------------------------
# Hrana/libSQL transient error retry
# -----------------------------
def _is_hrana_stale_stream_error(err: Exception) -> bool:
    s = str(err).lower()
    return ("hrana" in s and "404" in s and "stream not found" in s) or ("stream not found" in s)

def _exec_with_retry(engine: Engine, sql: str, params: Dict | Iterable[Dict] | None = None, *, tries: int = 2):
    """
    Execute a write (INSERT/UPDATE/DELETE) with a one-time retry on Hrana 'stream not found'.
    Accepts a single params dict or a list of dicts for executemany.
    Returns the result proxy of the final attempt.
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


# -----------------------------
# Computed Keywords — New rule-driven builder (A+B)
# -----------------------------
# (We keep your previous helpers below for backward compatibility,
#  but the hot path now uses build_computed_keywords.)

# ==== BEGIN: CKW Builder (INSERT near helpers) ====
import unicodedata

_CORP_SUFFIXES = {"llc","inc","co","ltd","llp","pllc","pc"}
_STOPWORDS = {"the","and","of","a","an","&"}

def _ckw_clean(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii","ignore").decode()
    s = s.lower()
    s = re.sub(r"[^\w\s\-]", " ", s)      # keep letters/digits/space/hyphen
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _ckw_tokens_from_name(name: str) -> list[str]:
    toks = [t for t in _ckw_clean(name).split(" ") if t and len(t) > 2]
    return [t for t in toks if t not in _CORP_SUFFIXES and t not in _STOPWORDS]

def _ckw_space_hyphen_variants(arr: list[str]) -> list[str]:
    out = []
    for t in arr:
        out.append(t)
        if " " in t:
            out.append(t.replace(" ","-"))
        if "-" in t:
            out.append(t.replace("-"," "))
    return out

def _ckw_plural_variants(arr: list[str]) -> list[str]:
    out = []
    for t in arr:
        out.append(t)
        if len(t) >= 3 and not t.endswith("s"):
            out.append(t + "s")
            if t.endswith(("s","x")) or t.endswith(("ch","sh")):
                out.append(t + "es")
    return out

@lru_cache(maxsize=512)
def _seed_for(category: str, service: str, rules_key: str) -> list[str]:
    """Fetch seeds from rules; rules_key ties cache to current rules digest."""
    rules = _load_ckw_rules()
    seeds = []
    if isinstance(rules.get("General"), dict):
        seeds += list(map(_ckw_clean, rules["General"].get(service, [])))
    seeds += list(map(_ckw_clean, rules.get(service, [])))
    seeds += list(map(_ckw_clean, rules.get(category, [])))
    return [t for t in seeds if t]

def build_computed_keywords(category: str, service: str, business_name: str,
                            max_terms: int | None = None) -> str:
    rules = _load_ckw_rules()
    maxN = int(max_terms or CKW_MAX_TERMS)
    # Global first (keeps them prominent), then seeds for (cat, svc), then name-signal
    base = list(map(_ckw_clean, rules.get("_global", [])))
    # rules_key toggles the LRU cache when rules change
    rules_key = str(hash(json.dumps(rules, sort_keys=True)))
    base += _seed_for(_ckw_clean(category), _ckw_clean(service), rules_key)
    name_bits = _ckw_tokens_from_name(business_name)
    # Avoid obvious dup: if service word equals a name token, keep only one
    base += [t for t in name_bits if t not in base]

    # Normalize, stopwords, then limited expansions with early cap
    arr = [t for t in base if t and t not in _STOPWORDS]
    arr = _ckw_space_hyphen_variants(arr)
    arr = _ckw_plural_variants(arr)

    out, seen = [], set()
    for t in arr:
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= maxN:
            break
    return ", ".join(out)
# ==== END: CKW Builder ====

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

# ==== BEGIN: ensure_schema per-statement retry (corrected to call retry helper with Engine) ====
def ensure_schema(engine: Engine) -> None:
    """
    Create/upgrade schema with per-statement retry to tolerate transient Hrana drops.
    Each DDL/DML runs independently via _exec_with_retry(engine, ...).
    Idempotent: safe to run on every boot.
    """
    # DDL statements (idempotent)
    ddls: list[str] = [
        # Tables
        """
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS services (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS vendors (
            id INTEGER PRIMARY KEY,
            category TEXT NOT NULL DEFAULT '',
            service TEXT NOT NULL DEFAULT '',
            business_name TEXT NOT NULL DEFAULT '',
            contact_name TEXT NOT NULL DEFAULT '',
            phone TEXT NOT NULL DEFAULT '',
            address TEXT NOT NULL DEFAULT '',
            website TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            keywords TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT '',
            updated_by TEXT NOT NULL DEFAULT '',
            computed_keywords TEXT NOT NULL DEFAULT ''
        )
        """,

        # Indexes (create if not exists)
        "CREATE INDEX IF NOT EXISTS idx_vendors_cat        ON vendors(category)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_svc        ON vendors(service)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_bus        ON vendors(business_name)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_kw         ON vendors(keywords)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_phone      ON vendors(phone)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_cat_lower  ON vendors(LOWER(category))",
        "CREATE INDEX IF NOT EXISTS idx_vendors_svc_lower  ON vendors(LOWER(service))",
        "CREATE INDEX IF NOT EXISTS idx_vendors_bus_lower  ON vendors(LOWER(business_name))",
        "CREATE INDEX IF NOT EXISTS idx_vendors_ckw        ON vendors(computed_keywords)"
    ]

    # Optional one-time fixes (idempotent backfills/sanitizers)
    fixes: list[str] = [
        # backfill timestamps if blanks
        """
        UPDATE vendors
           SET created_at = COALESCE(NULLIF(created_at, ''), updated_at, strftime('%Y-%m-%dT%H:%M:%SZ','now')),
               updated_at = COALESCE(NULLIF(updated_at, ''), strftime('%Y-%m-%dT%H:%M:%SZ','now'))
         WHERE (created_at = '' OR updated_at = '')
        """,
        # normalize NULLs to empty string for text fields (safety)
        """
        UPDATE vendors
           SET category      = COALESCE(category,''),
               service       = COALESCE(service,''),
               business_name = COALESCE(business_name,''),
               contact_name  = COALESCE(contact_name,''),
               phone         = COALESCE(phone,''),
               address       = COALESCE(address,''),
               website       = COALESCE(website,''),
               notes         = COALESCE(notes,''),
               keywords      = COALESCE(keywords,''),
               updated_by    = COALESCE(updated_by,''),
               computed_keywords = COALESCE(computed_keywords,'')
         WHERE 1=1
        """
    ]

    # Run DDLs with per-statement retry
    for stmt in ddls:
        _exec_with_retry(engine, stmt)

    # Run fixes with per-statement retry
    for stmt in fixes:
        _exec_with_retry(engine, stmt)
# ==== END: ensure_schema per-statement retry (corrected) ====

# ==== BEGIN: CKW Schema Ensure (C) ====
def _ensure_ckw_columns(engine: Engine) -> None:
    """
    Adds columns if they don't exist:
      computed_keywords TEXT (already in base DDL, but safe-check)
      ckw_locked INTEGER DEFAULT 0
      ckw_version TEXT
    Uses per-statement try/ignore to avoid long transactions.
    """
    with engine.connect() as cx:
        cols = {r[1] for r in cx.execute(sql_text("PRAGMA table_info(vendors)")).fetchall()}
    add_sql = []
    if "computed_keywords" not in cols:
        add_sql.append("ALTER TABLE vendors ADD COLUMN computed_keywords TEXT")
    if "ckw_locked" not in cols:
        add_sql.append("ALTER TABLE vendors ADD COLUMN ckw_locked INTEGER DEFAULT 0")
    if "ckw_version" not in cols:
        add_sql.append("ALTER TABLE vendors ADD COLUMN ckw_version TEXT")
    for stmt in add_sql:
        try:
            _exec_with_retry(engine, stmt)
        except Exception:
            pass  # safe if concurrent or already added
# ==== END: CKW Schema Ensure ====

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
    with engine.connect() as conn:  # read-only
        df = pd.read_sql(sql_text("SELECT * FROM vendors ORDER BY lower(business_name)"), conn)

    for col in [
        "contact_name",
        "phone",
        "address",
        "website",
        "notes",
        "keywords",
        "computed_keywords",
        "ckw_locked",
        "ckw_version",
        "service",
        "created_at",
        "updated_at",
        "updated_by",
    ]:
        if col not in df.columns:
            df[col] = ""

    df["phone_fmt"] = df["phone"].apply(_format_phone)
    return df

# ---- cached taxonomy lookups (no Engine arg; safe to cache) ----
@st.cache_data(ttl=60, show_spinner=False)
def list_names(table: str) -> list[str]:
    with ENGINE.connect() as conn:  # read-only
        rows = conn.execute(sql_text(f"SELECT name FROM {table} ORDER BY lower(name)")).fetchall()
    return [r[0] for r in rows]

def usage_count(engine: Engine, col: str, name: str) -> int:
    with engine.connect() as conn:  # read-only
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
        conn.execute(sql_text("DELETE FROM services WHERE name=:old"), {"name": old})

def delete_service_with_reassign(engine: Engine, tgt: str, repl: str) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("UPDATE vendors SET service=:repl WHERE service=:tgt"), {"repl": repl, "tgt": tgt})
        conn.execute(sql_text("DELETE FROM services WHERE name=:tgt"), {"tgt": tgt})

# Ensure schema on boot
ensure_schema(ENGINE)
# Ensure CKW columns on boot (A–C requirement)
try:
    _ensure_ckw_columns(ENGINE)
except Exception as _e:
    st.warning(f"CKW column ensure skipped: {_e.__class__.__name__}: {_e}")

# -----------------------------
# Computed Keywords (legacy helpers kept for compatibility; hot path uses new builder)
# -----------------------------
CKW_RULES: dict = _resolve_json("CKW_RULES", {})  # supports TOML table or JSON string

def _canon_tokenize(text: str) -> list[str]:
    if not text:
        return []
    s = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).lower()
    toks = [t.strip() for t in s.split() if t.strip()]
    _STOPWORDS_LEG = {
        "the","and","inc","llc","ltd","co","corp","company","services","service","of","for","to","in","on","at","a","an"
    }
    return [t for t in toks if len(t) > 2 and t not in _STOPWORDS_LEG]

def _rules_for_pair(rules: dict, category: str, service: str) -> list[str]:
    cat = (category or "").strip().lower()
    svc = (service or "").strip().lower()
    out: list[str] = []
    if not isinstance(rules, dict):
        return out
    if cat and svc and cat in rules and isinstance(rules[cat], dict):
        svcd = rules[cat].get(svc)
        if isinstance(svcd, list):
            out.extend([str(x).strip().lower() for x in svcd if str(x).strip()])
    if cat in rules and isinstance(rules[cat], list):
        out.extend([str(x).strip().lower() for x in rules[cat] if str(x).strip()])
    if "_global" in rules and isinstance(rules["_global"], list):
        out.extend([str(x).strip().lower() for x in rules["_global"] if str(x).strip()])
    seen = set(); uniq = []
    for t in out:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq

def _join_kw(terms: list[str], max_terms: int = 16) -> str:
    seen=set(); out=[]
    for t in terms:
        tt = t.strip().lower()
        if not tt or tt in seen:
            continue
        seen.add(tt); out.append(tt)
        if len(out) >= max_terms:
            break
    return ", ".join(out)

def _kw_from_row_fast(row: dict) -> str:
    seeds = _rules_for_pair(CKW_RULES, row.get("category") or "", row.get("service") or "")
    explicit = [t.strip().lower() for t in (row.get("keywords") or "").split(",") if t.strip()]
    base = " ".join([
        str(row.get("business_name") or ""),
        str(row.get("notes") or ""),
        str(row.get("service") or ""),
        str(row.get("category") or ""),
        str(row.get("address") or ""),
    ])
    toks = _canon_tokenize(base)
    terms = seeds + explicit + toks
    return _join_kw(terms)

# ==== BEGIN: helper – priority search using computed_keywords first ====
def _filter_and_rank_by_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return df
    q = str(query).strip().lower()
    ckw = df.get("computed_keywords", pd.Series([""] * len(df), index=df.index)).astype(str).str.lower()
    other = pd.concat([
        df.get("business_name", ""),
        df.get("category", ""),
        df.get("service", ""),
        df.get("contact_name", ""),
        df.get("phone", ""),
        df.get("address", ""),
        df.get("website", ""),
        df.get("notes", ""),
        df.get("keywords", ""),
    ], axis=1).astype(str).agg(" ".join, axis=1).str.lower()
    hit_ckw = ckw.str.contains(q, na=False)
    hit_oth = other.str.contains(q, na=False)
    any_hit_mask = hit_ckw | hit_oth
    dfm = df.loc[any_hit_mask].copy()
    dfm["_rank_ckw"] = (~hit_ckw.loc[dfm.index]).astype(int)
    dfm.sort_values(
        by=["_rank_ckw", "business_name"],
        key=lambda s: s.astype(str),
        inplace=True,
        kind="mergesort",
    )
    dfm.drop(columns=["_rank_ckw"], inplace=True, errors="ignore")
    return dfm
# ==== END: helper – priority search using computed_keywords first ====

# -----------------------------
# Form state helpers (Add / Edit / Delete)
# -----------------------------
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
    st.session_state["_pending_add_reset"] = True

EDIT_FORM_KEYS = [
    "edit_vendor_id", "edit_business_name", "edit_category", "edit_service",
    "edit_contact_name", "edit_phone", "edit_address", "edit_website",
    "edit_notes", "edit_keywords", "edit_row_updated_at", "edit_last_loaded_id",
    # new ephemeral keys for CKW UI
    "_ckw_suggest"
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
        "_ckw_suggest": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("edit_form_version", 0)
    st.session_state.setdefault("_pending_edit_reset", False)
    st.session_state.setdefault("edit_last_done", None)
    st.session_state.setdefault("edit_nonce", uuid.uuid4().hex)

def _apply_edit_reset_if_needed():
    if st.session_state.get("_pending_edit_reset"):
        for k in EDIT_FORM_KEYS:
            if k in ("edit_vendor_id", "edit_row_updated_at", "edit_last_loaded_id"):
                st.session_state[k] = None
            else:
                st.session_state[k] = ""
        if "edit_provider_label" in st.session_state:
            del st.session_state["edit_provider_label"]
        st.session_state["_pending_edit_reset"] = False
        st.session_state["edit_form_version"] += 1

def _queue_edit_form_reset():
    st.session_state["_pending_edit_reset"] = True

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
# UI
# -----------------------------
_tab_names = [
    "Browse Providers",
    "Add / Edit / Delete Provider",
    "Category Admin",
    "Service Admin",
    "Maintenance",
]
if _SHOW_DEBUG:
    _tab_names.append("Debug")

_tabs = st.tabs(_tab_names)

# ---------- Browse
with _tabs[0]:
    df = load_df(ENGINE)

    # --- Search input at 25% width (table remains full width) ---
    left, right = st.columns([1, 3])  # 25% / 75% split for this row only
    with left:
        q = st.text_input(
            "Search",
            placeholder="Search providers… (press Enter)",
            label_visibility="collapsed",
            key="q",
        )
    # Keep ?q= in the URL for shareable links (best-effort)
    try:
        st.query_params["q"] = st.session_state.get("q", "")
    except Exception:
        pass

    # Prefer matches in computed_keywords first, then other fields
    filtered = _filter_and_rank_by_query(df, (st.session_state.get("q") or ""))

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
        "computed_keywords",
    ]

    vdf = filtered[view_cols].rename(columns={"phone_fmt": "phone"})

    # Read-only table with clickable website links
    st.dataframe(
        vdf,
        use_container_width=True,
        hide_index=True,
        column_config={
            "business_name": st.column_config.TextColumn("Provider"),
            "website": st.column_config.LinkColumn("Website"),
            "notes": st.column_config.TextColumn(width=420),
            "keywords": st.column_config.TextColumn(width=260),
            "computed_keywords": st.column_config.TextColumn(width=260),
        },
    )

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    st.download_button(
        "Download filtered view (CSV)",
        data=vdf.to_csv(index=False).encode("utf-8"),
        file_name=f"providers_{ts}.csv",
        mime="text/csv",
    )

# ---------- Add/Edit/Delete Provider
with _tabs[1]:
    # ===== Add Provider =====
    st.subheader("Add Provider")
    _init_add_form_defaults()
    _apply_add_reset_if_needed()  # apply queued reset BEFORE creating widgets

    cats = list_names("categories")
    servs = list_names("services")

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

        submitted = st.form_submit_button("Add Provider")

    if submitted:
        add_nonce = _nonce("add")
        if st.session_state.get("add_last_done") == add_nonce:
            st.info("Add already processed.")
            st.stop()

        business_name = (st.session_state["add_business_name"] or "").strip()
        category      = (st.session_state["add_category"] or "").strip()
        service       = (st.session_state["add_service"] or "").strip(" /;,")
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
            st.error("Provider and Category are required.")
        else:
            try:
                now = datetime.utcnow().isoformat(timespec="seconds")
                # === NEW: compute once per Save using rule-driven builder (D) ===
                suggested_ckw = build_computed_keywords(category, service, business_name, CKW_MAX_TERMS)
                ckw_final = suggested_ckw
                ckw_locked = 0  # auto-generated on Add

                _exec_with_retry(
                    ENGINE,
                    """
                    INSERT INTO vendors(category, service, business_name, contact_name, phone, address,
                                        website, notes, keywords, computed_keywords, ckw_locked, ckw_version,
                                        created_at, updated_at, updated_by)
                    VALUES(:category, NULLIF(:service, ''), :business_name, :contact_name, :phone, :address,
                           :website, :notes, :keywords, :computed_keywords, :ckw_locked, :ckw_version,
                           :now, :now, :user)
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
                        "computed_keywords": ckw_final,
                        "ckw_locked": int(ckw_locked),
                        "ckw_version": CKW_VERSION,
                        "now": now,
                        "user": _updated_by(),
                    },
                )
                st.session_state["add_last_done"] = add_nonce
                st.success(f"Provider added: {business_name}")
                _queue_add_form_reset()
                list_names.clear()  # refresh cached taxonomy
                _nonce_rotate("add")
                st.rerun()
            except Exception as e:
                st.error(f"Add failed: {e}")

    st.divider()
    st.subheader("Edit Provider")

    df_all = load_df(ENGINE)

    if df_all.empty:
        st.info("No providers yet. Use 'Add Provider' above to create your first record.")
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
                    "_ckw_suggest": None,
                })

        # -------- Edit form --------
        edit_form_key = f"edit_vendor_form_{st.session_state['edit_form_version']}"
        with st.form(edit_form_key, clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Provider *", key="edit_business_name")

                cats = list_names("categories")
                servs = list_names("services")

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

                # ==== BEGIN: CKW Edit Suggest (E) ====
                st.caption("Computed keywords are used for search in the read-only app.")
                ckw_current = st.text_area(
                    "computed_keywords (editable)",
                    value=(id_to_row.get(int(st.session_state["edit_vendor_id"]), {}).get("computed_keywords", "") if st.session_state["edit_vendor_id"] else ""),
                    height=80,
                    key="edit_computed_keywords",
                )
                if st.form_submit_button("Suggest (recompute)", type="secondary"):
                    cat = (st.session_state["edit_category"] or "").strip()
                    svc = (st.session_state["edit_service"] or "").strip(" /;,")
                    bn  = (st.session_state["edit_business_name"] or "").strip()
                    st.session_state["_ckw_suggest"] = build_computed_keywords(cat, svc, bn, CKW_MAX_TERMS)
                if st.session_state.get("_ckw_suggest"):
                    st.info("Suggested: " + st.session_state["_ckw_suggest"])
                # ==== END: CKW Edit Suggest (E) ====

            edited = st.form_submit_button("Save Changes")

        if edited:
            edit_nonce = _nonce("edit")
            if st.session_state.get("edit_last_done") == edit_nonce:
                st.info("Edit already processed.")
                st.stop()

            vid = st.session_state.get("edit_vendor_id")
            if vid is None:
                st.error("Select a provider first.")
            else:
                bn  = (st.session_state["edit_business_name"] or "").strip()
                cat = (st.session_state["edit_category"] or "").strip()
                svc = (st.session_state["edit_service"] or "").strip(" /;,")
                phone_norm = _normalize_phone(st.session_state["edit_phone"])
                if phone_norm and len(phone_norm) != 10:
                    st.error("Phone must be 10 digits or blank.")
                elif not bn or not cat:
                    st.error("Provider and Category are required.")
                else:
                    try:
                        prev_updated = st.session_state.get("edit_row_updated_at") or ""
                        now = datetime.utcnow().isoformat(timespec="seconds")

                        # Decide lock: if user edited away from suggestion, lock; else auto (0)
                        ckw_current_val = (st.session_state.get("edit_computed_keywords") or "").strip()
                        suggested_now   = st.session_state.get("_ckw_suggest") or build_computed_keywords(cat, svc, bn, CKW_MAX_TERMS)
                        ckw_locked_val  = 1 if (ckw_current_val and ckw_current_val.strip() != suggested_now.strip()) else 0

                        res = _exec_with_retry(ENGINE, """
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
                                   computed_keywords=:ckw,
                                   ckw_locked=:ckw_locked,
                                   ckw_version=:ckw_version,
                                   updated_at=:now,
                                   updated_by=:user
                             WHERE id=:id AND (updated_at=:prev_updated OR :prev_updated='')
                        """, {
                            "category": cat,
                            "service": svc,
                            "business_name": bn,
                            "contact_name": (st.session_state["edit_contact_name"] or "").strip(),
                            "phone": phone_norm,
                            "address": (st.session_state["edit_address"] or "").strip(),
                            "website": _sanitize_url(st.session_state["edit_website"]),
                            "notes": (st.session_state["edit_notes"] or "").strip(),
                            "keywords": (st.session_state["edit_keywords"] or "").strip(),
                            "ckw": ckw_current_val or suggested_now,
                            "ckw_locked": int(ckw_locked_val),
                            "ckw_version": CKW_VERSION,
                            "now": now, "user": _updated_by(),
                            "id": int(vid),
                            "prev_updated": prev_updated,
                        })
                        rowcount = res.rowcount or 0

                        if rowcount == 0:
                            st.warning("No changes applied (stale selection or already updated). Refresh and try again.")
                        else:
                            st.session_state["edit_last_done"] = edit_nonce
                            st.success(f"Provider updated: {bn}")
                            _queue_edit_form_reset()
                            _nonce_rotate("edit")
                            list_names.clear()
                            st.rerun()
                    except Exception as e:
                        st.error(f"Update failed: {e}")

        st.markdown("---")
        st.subheader("Delete Provider")

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
            deleted = st.form_submit_button("Delete Provider")

        if deleted:
            del_nonce = _nonce("delete")
            if st.session_state.get("delete_last_done") == del_nonce:
                st.info("Delete already processed.")
                st.stop()

            vid = st.session_state.get("delete_vendor_id")
            if vid is None:
                st.error("Select a provider first.")
            else:
                try:
                    row = df_all.loc[df_all["id"] == int(vid)]
                    prev_updated = (row.iloc[0]["updated_at"] if not row.empty else "") or ""
                    res = _exec_with_retry(ENGINE, """
                        DELETE FROM vendors
                         WHERE id=:id AND (updated_at=:prev_updated OR :prev_updated='')
                    """, {"id": int(vid), "prev_updated": prev_updated})
                    rowcount = res.rowcount or 0

                    if rowcount == 0:
                        st.warning("No delete performed (stale selection). Refresh and try again.")
                    else:
                        st.session_state["delete_last_done"] = del_nonce
                        st.success("Provider deleted.")
                        _queue_delete_form_reset()
                        _nonce_rotate("delete")
                        list_names.clear()
                        st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")

# ---------- Category Admin
with _tabs[2]:
    st.caption("Category is required. Manage the reference list and reassign providers safely.")
    _init_cat_defaults()
    _apply_cat_reset_if_needed()

    cats = list_names("categories")
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
                    _exec_with_retry(ENGINE, "INSERT OR IGNORE INTO categories(name) VALUES(:n)", {"n": new_cat.strip()})
                    list_names.clear()
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
                        rename_category_and_cascade(ENGINE, old, new.strip())
                        list_names.clear()
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
                cnt = usage_count(ENGINE, "category", tgt)
                st.write(f"In use by {cnt} provider(s).")
                if cnt == 0:
                    if st.button("Delete category (no usage)", key="cat_del_btn"):
                        try:
                            _exec_with_retry(ENGINE, "DELETE FROM categories WHERE name=:n", {"n": tgt})
                            list_names.clear()
                            st.success("Deleted.")
                            _queue_cat_reset()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete category failed: {e}")
                else:
                    repl_options = ["— Select —"] + [c for c in cats if c != tgt]
                    repl = st.selectbox("Reassign providers to…", options=repl_options, key="cat_reassign_to")
                    if st.button("Reassign providers then delete", key="cat_reassign_btn"):
                        if repl == "— Select —":
                            st.error("Choose a category to reassign to.")
                        else:
                            try:
                                delete_category_with_reassign(ENGINE, tgt, repl)
                                list_names.clear()
                                st.success("Reassigned and deleted.")
                                _queue_cat_reset()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Reassign+delete failed: {e}")

# ---------- Service Admin
with _tabs[3]:
    st.caption("Service is optional on providers. Manage the reference list here.")
    _init_svc_defaults()
    _apply_svc_reset_if_needed()

    servs = list_names("services")
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
                    _exec_with_retry(ENGINE, "INSERT OR IGNORE INTO services(name) VALUES(:n)", {"n": new_s.strip()})
                    list_names.clear()
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
                        rename_service_and_cascade(ENGINE, old, new.strip())
                        list_names.clear()
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
                cnt = usage_count(ENGINE, "service", tgt)
                st.write(f"In use by {cnt} provider(s).")
                if cnt == 0:
                    if st.button("Delete service (no usage)", key="svc_del_btn"):
                        try:
                            _exec_with_retry(ENGINE, "DELETE FROM services WHERE name=:n", {"n": tgt})
                            list_names.clear()
                            st.success("Deleted.")
                            _queue_svc_reset()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete service failed: {e}")
                else:
                    repl_options = ["— Select —"] + [s for s in servs if s != tgt]
                    repl = st.selectbox("Reassign providers to…", options=repl_options, key="svc_reassign_to")
                    if st.button("Reassign providers then delete service", key="svc_reassign_btn"):
                        if repl == "— Select —":
                            st.error("Choose a service to reassign to.")
                        else:
                            try:
                                delete_service_with_reassign(ENGINE, tgt, repl)
                                list_names.clear()
                                st.success("Reassigned and deleted.")
                                _queue_svc_reset()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Reassign+delete service failed: {e}")

# ---------- Maintenance
with _tabs[4]:
    st.caption("One-click cleanups and keyword recompute tools.")

    # ==== BEGIN: Maintenance ▸ Integrity Self-Test (kept) ====
    with st.expander("Integrity Self-Test", expanded=False):
        st.caption("Runs PRAGMA checks and basic counts. Read-only; safe anytime.")
        if st.button("Run checks", type="primary"):
            results: Dict[str, Any] = {}
            try:
                with ENGINE.connect() as cx:  # read-only ops
                    quick = cx.exec_driver_sql("PRAGMA quick_check").scalar()
                    results["quick_check"] = quick
                    integ = cx.exec_driver_sql("PRAGMA integrity_check").scalar()
                    results["integrity_check"] = integ
                    counts = {}
                    for tbl in ("vendors", "categories", "services"):
                        try:
                            c = cx.exec_driver_sql(f"SELECT COUNT(*) FROM {tbl}").scalar()
                            counts[tbl] = int(c or 0)
                        except Exception as e:
                            counts[tbl] = f"error: {type(e).__name__}: {e}"
                    results["counts"] = counts
                    ix = cx.exec_driver_sql("PRAGMA index_list('vendors')").mappings().all()
                    results["vendors_indexes"] = [
                        {"seq": int(r.get("seq", 0)), "name": r.get("name"), "unique": bool(r.get("unique", 0))}
                        for r in ix
                    ]
                ok = (str(results.get("quick_check", "")).lower() == "ok") and (
                    str(results.get("integrity_check", "")).lower() == "ok"
                )
                (st.success if ok else st.error)(
                    f"Integrity {'OK' if ok else 'issues detected'} — see details below."
                )
                st.json(results)
            except Exception as e:
                st.error(f"Integrity test failed: {type(e).__name__}: {e}")
    # ==== END: Maintenance ▸ Integrity Self-Test ====

    # ====== Computed Keywords tools (F) ======
    st.subheader("Computed Keywords")

    with st.expander("Recompute (versioned, unlocked only)", expanded=False):
        st.write("Recompute for rows where (ckw_locked=0) and (ckw_version != CKW_VERSION or empty).")
        do_run = st.button("Recompute all (unlocked & stale)")
        if do_run:
            try:
                with ENGINE.connect() as cx:
                    rows = cx.execute(sql_text("""
                        SELECT id, category, service, business_name
                        FROM vendors
                        WHERE IFNULL(ckw_locked,0)=0
                          AND (ckw_version IS NULL OR ckw_version <> :v OR computed_keywords IS NULL OR TRIM(computed_keywords)='')
                    """), {"v": CKW_VERSION}).fetchall()
                total = len(rows)
                if total == 0:
                    st.success("No rows need recompute.")
                else:
                    batch, done = [], 0
                    for r in rows:
                        ckw = build_computed_keywords(r.service, r.service, r.business_name, CKW_MAX_TERMS)  # service twice? No—fix:
                        # Correct build:
                        ckw = build_computed_keywords(r.category, r.service, r.business_name, CKW_MAX_TERMS)
                        batch.append({"id": r.id, "ckw": ckw, "ver": CKW_VERSION})
                        if len(batch) >= 300:
                            _exec_with_retry(ENGINE,
                                "UPDATE vendors SET computed_keywords=:ckw, ckw_version=:ver WHERE id=:id",
                                batch)
                            done += len(batch); batch.clear()
                            st.write(f"Updated {done}/{total}…")
                    if batch:
                        _exec_with_retry(ENGINE,
                            "UPDATE vendors SET computed_keywords=:ckw, ckw_version=:ver WHERE id=:id",
                            batch)
                        done += len(batch)
                    st.success(f"Recomputed {done} row(s).")
            except Exception as e:
                st.error(f"Bulk recompute failed: {e.__class__.__name__}: {e}")

    st.divider()
    st.subheader("Export / Import")

    # Export full, untruncated CSV of all columns/rows
    query = "SELECT * FROM vendors ORDER BY lower(business_name)"
    with ENGINE.connect() as conn:  # read-only
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
            "Export all providers (formatted phones)",
            data=full_formatted.to_csv(index=False).encode("utf-8"),
            file_name=f"providers_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.csv",
            mime="text/csv",
        )
    with colB:
        st.download_button(
            "Export all providers (digits-only phones)",
            data=full.to_csv(index=False).encode("utf-8"),
            file_name=f"providers_raw_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.csv",
            mime="text/csv",
        )

    # CSV Restore UI (Append-only, ID-checked) — helpers bundled
    have_csv_helpers = all(name in globals() for name in ("_prepare_csv_for_append", "_execute_append_only"))
    if have_csv_helpers:
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
                        ENGINE,
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
                            inserted = _execute_append_only(ENGINE, with_id_df, without_id_df, insertable_cols)
                            st.success(f"Inserted {inserted} row(s). Rejected existing id(s): {rejected_ids or 'None'}")
                except Exception as e:
                    st.error(f"CSV restore failed: {e}")
    else:
        st.info("CSV Restore is disabled in this build (helpers not bundled).")

    st.divider()
    st.subheader("Integrity self-test")

    if st.button("Run insert/rollback self-test"):
        try:
            conn = ENGINE.connect()
            trans = conn.begin()  # manual transaction so we can rollback
            now = datetime.utcnow().isoformat(timespec="seconds")
            probe_name = f"_probe_vendor_{uuid.uuid4().hex[:8]}"
            conn.execute(
                sql_text("""
                    INSERT INTO vendors(category, service, business_name, contact_name, phone, address,
                                        website, notes, keywords, computed_keywords, ckw_locked, ckw_version,
                                        created_at, updated_at, updated_by)
                    VALUES(:category, :service, :business_name, :contact_name, :phone, :address,
                           :website, :notes, :keywords, :computed_keywords, :ckw_locked, :ckw_version,
                           :created_at, :updated_at, :updated_by)
                """),
                {
                    "category": "_Probe",
                    "service": None,
                    "business_name": probe_name,
                    "contact_name": "Tester",
                    "phone": "5555555555",
                    "address": "123 Test Ln",
                    "website": "https://example.com",
                    "notes": "self-test insert",
                    "keywords": "probe, test",
                    "computed_keywords": build_computed_keywords("_Probe", "", probe_name, CKW_MAX_TERMS),
                    "ckw_locked": 0,
                    "ckw_version": CKW_VERSION,
                    "created_at": now,
                    "updated_at": now,
                    "updated_by": _updated_by(),
                }
            )
            exists = conn.execute(
                sql_text("SELECT COUNT(*) AS c FROM vendors WHERE business_name=:n"),
                {"n": probe_name}
            ).scalar() or 0
            trans.rollback()
            conn.close()

            if exists:
                st.success("Self-test passed: insert was visible inside transaction and rolled back cleanly.")
            else:
                st.warning("Self-test inconclusive: inserted row not visible during transaction.")
        except Exception as e:
            st.error(f"Self-test failed: {type(e).__name__}: {e}")

    st.divider()
    st.subheader("Data cleanup")

    if st.button("Normalize phone numbers & Title Case (providers + categories/services)"):
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
            with ENGINE.connect() as conn:  # read for list; writes use retry below
                rows = conn.execute(sql_text("SELECT * FROM vendors")).fetchall()
            for r in rows:
                row = dict(r._mapping) if hasattr(r, "_mapping") else dict(r)
                pid = int(row["id"])

                vals = {c: to_title(row.get(c)) for c in TEXT_COLS_TO_TITLE}
                vals["website"] = _sanitize_url((row.get("website") or "").strip())
                vals["phone"] = _normalize_phone(row.get("phone") or "")
                vals["id"] = pid

                _exec_with_retry(
                    ENGINE,
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
                    """,
                    vals,
                )
                changed_vendors += 1

            with ENGINE.connect() as conn:
                cat_rows = conn.execute(sql_text("SELECT name FROM categories")).fetchall()
            for (old_name,) in cat_rows:
                new_name = to_title(old_name)
                if new_name != old_name:
                    _exec_with_retry(ENGINE, "INSERT OR IGNORE INTO categories(name) VALUES(:n)", {"n": new_name})
                    _exec_with_retry(
                        ENGINE,
                        "UPDATE vendors SET category=:new WHERE category=:old",
                        {"new": new_name, "old": old_name},
                    )
                    _exec_with_retry(ENGINE, "DELETE FROM categories WHERE name=:old", {"old": old_name})

            with ENGINE.connect() as conn:
                svc_rows = conn.execute(sql_text("SELECT name FROM services")).fetchall()
            for (old_name,) in svc_rows:
                new_name = to_title(old_name)
                if new_name != old_name:
                    _exec_with_retry(ENGINE, "INSERT OR IGNORE INTO services(name) VALUES(:n)", {"n": new_name})
                    _exec_with_retry(
                        ENGINE,
                        "UPDATE vendors SET service=:new WHERE service=:old",
                        {"new": new_name, "old": old_name},
                    )
                    _exec_with_retry(ENGINE, "DELETE FROM services WHERE name=:old", {"old": old_name})

            st.success(f"Providers normalized: {changed_vendors}. Categories/services retitled and reconciled.")
            list_names.clear()
            _queue_cat_reset()
            _queue_svc_reset()
            st.rerun()
        except Exception as e:
            st.error(f"Normalization failed: {e}")

    if st.button("Backfill created_at/updated_at when missing"):
        try:
            now = datetime.utcnow().isoformat(timespec="seconds")
            _exec_with_retry(
                ENGINE,
                """
                UPDATE vendors
                   SET created_at = CASE WHEN created_at IS NULL OR created_at = '' THEN :now ELSE created_at END,
                       updated_at = CASE WHEN updated_at IS NULL OR updated_at = '' THEN :now ELSE updated_at END
                """,
                {"now": now},
            )
            st.success("Backfill complete.")
        except Exception as e:
            st.error(f"Backfill failed: {e}")

    if st.button("Trim whitespace in text fields (safe)"):
        try:
            changed = 0
            with ENGINE.connect() as conn:
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
                    "notes": clean_soft(r[7]),
                    "keywords": clean_soft(r[8]),
                    "phone": r[9],
                    "id": pid,
                }
                _exec_with_retry(
                    ENGINE,
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
                    """,
                    vals,
                )
                changed += 1
            st.success(f"Whitespace trimmed on {changed} row(s).")
        except Exception as e:
            st.error(f"Trim failed: {e}")

    with st.expander("Storage maintenance (optional)"):
        c1, c2 = st.columns(2)
        if c1.button("ANALYZE"):
            try:
                with ENGINE.connect() as cx:
                    cx.exec_driver_sql("ANALYZE")
                st.success("ANALYZE completed.")
            except Exception as e:
                st.error(f"ANALYZE failed: {e}")
        if c2.button("VACUUM"):
            try:
                with ENGINE.connect() as cx:
                    cx.exec_driver_sql("VACUUM")
                st.success("VACUUM completed.")
            except Exception as e:
                st.error(f"VACUUM failed: {e}")

# ---------- Debug (only when enabled)
if _SHOW_DEBUG:
    with _tabs[-1]:
        engine_info: Dict = _DB_DBG
        st.subheader("Status & Secrets (debug)")
        st.json(engine_info)

        with ENGINE.connect() as conn:  # read-only
            vendors_cols = conn.execute(sql_text("PRAGMA table_info(vendors)")).fetchall()
            categories_cols = conn.execute(sql_text("PRAGMA table_info(categories)")).fetchall()
            services_cols = conn.execute(sql_text("PRAGMA table_info(services)")).fetchall()

            idx_rows = conn.execute(sql_text("PRAGMA index_list(vendors)")).fetchall()
            vendors_indexes = [
                {"seq": r[0], "name": r[1], "unique": bool(r[2]), "origin": r[3], "partial": bool(r[4])} for r in idx_rows
            ]

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

        def _quick_probe_replica_and_indexes(engine: Engine) -> dict:
            out: dict = {}
            try:
                with engine.connect() as cx:
                    out["select_1"] = cx.execute(sql_text("SELECT 1")).scalar_one()
                    out["vendors_count"] = cx.execute(sql_text("SELECT COUNT(*) FROM vendors")).scalar_one()
                    out["ckw_nonblank"] = cx.execute(sql_text("""
                        SELECT COUNT(*) FROM vendors WHERE TRIM(COALESCE(computed_keywords,'')) <> ''
                    """)).scalar_one()
                    out["max_updated_at"] = cx.execute(sql_text("""
                        SELECT COALESCE(MAX(updated_at), '') FROM vendors
                    """)).scalar_one()
                    rows = cx.execute(sql_text("PRAGMA index_list('vendors')")).all()
                    out["indexes"] = [{"seq": r[0], "name": r[1], "unique": bool(r[2])} for r in rows]
            except Exception as e:
                out["error"] = f"{e.__class__.__name__}: {e}"
            return out

        with st.expander("Quick Probe: replica + indexes + ckw + timestamps"):
            try:
                probe = _quick_probe_replica_and_indexes(ENGINE)
                st.json(probe)
            except Exception as e:
                st.error(f"Probe failed: {e.__class__.__name__}: {e}")
