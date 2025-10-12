# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import hmac
import time
import uuid
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# ---- register libsql dialect (must be AFTER "import streamlit as st") ----
try:
    import sqlalchemy_libsql  # ensures 'sqlite+libsql' dialect is registered
except Exception:
    pass
# ---- end dialect registration ----


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


# -----------------------------
# DB helpers
# -----------------------------
REQUIRED_VENDOR_COLUMNS: List[str] = ["business_name", "category"]  # service optional


def build_engine() -> Tuple[Engine, Dict]:
    """Prefer Turso/libsql embedded replica; otherwise local sqlite if FORCE_LOCAL=1."""
    info: Dict = {}

    url = (_resolve_str("TURSO_DATABASE_URL", "") or "").strip()
    token = (_resolve_str("TURSO_AUTH_TOKEN", "") or "").strip()
    embedded_path = os.path.abspath(_resolve_str("EMBEDDED_DB_PATH", "vendors-embedded.db") or "vendors-embedded.db")

    if not url:
        # No remote configured → plain local file DB
        eng = create_engine(
            "sqlite:///vendors.db",
            pool_pre_ping=True,
            pool_recycle=300,
            pool_reset_on_return="commit",
        )
        info.update(
            {
                "using_remote": False,
                "sqlalchemy_url": "sqlite:///vendors.db",
                "dialect": eng.dialect.name,
                "driver": getattr(eng.dialect, "driver", ""),
            }
        )
        return eng, info

    # Embedded replica: local file that syncs to your remote Turso DB
    try:
        # Normalize sync_url: embedded REQUIRES libsql:// (no sqlite+libsql, no ?secure=true)
        raw = url
        if raw.startswith("sqlite+libsql://"):
            host = raw.split("://", 1)[1].split("?", 1)[0]  # drop any ?secure=true
            sync_url = f"libsql://{host}"
        else:
            sync_url = raw.split("?", 1)[0]  # already libsql://...

        eng = create_engine(
            f"sqlite+libsql:///{embedded_path}",
            connect_args={
                "auth_token": token,
                "sync_url": sync_url,
            },
            pool_pre_ping=True,
            pool_recycle=300,
            pool_reset_on_return="commit",
        )
        with eng.connect() as c:
            c.exec_driver_sql("select 1;")

        info.update(
            {
                "using_remote": True,
                "strategy": "embedded_replica",
                "sqlalchemy_url": f"sqlite+libsql:///{embedded_path}",
                "dialect": eng.dialect.name,
                "driver": getattr(eng.dialect, "driver", ""),
                "sync_url": sync_url,
            }
        )
        return eng, info

    except Exception as e:
        info["remote_error"] = f"{e}"
        allow_local = _as_bool(os.getenv("FORCE_LOCAL"), False)
        if allow_local:
            eng = create_engine(
                "sqlite:///vendors.db",
                pool_pre_ping=True,
                pool_recycle=300,
                pool_reset_on_return="commit",
            )
            info.update(
                {
                    "using_remote": False,
                    "sqlalchemy_url": "sqlite:///vendors.db",
                    "dialect": eng.dialect.name,
                    "driver": getattr(eng.dialect, "driver", ""),
                }
            )
            return eng, info

        st.error("Remote DB unavailable and FORCE_LOCAL is not set. Aborting to protect data.")
        raise


def ensure_schema(engine: Engine, show_debug: bool = False) -> None:
    """
    Execute schema DDL safely, one statement at a time, using driver-level execution.
    Set show_debug=True (or SCHEMA_DEBUG=true in secrets/env) to print each statement.
    """
    # Base DDL (note: keep each CREATE ... as its own statement for clarity)
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
        "CREATE INDEX IF NOT EXISTS idx_vendors_phone ON vendors(phone)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_kw    ON vendors(keywords)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_bus   ON vendors(business_name)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_bus_lower ON vendors(lower(business_name))",
        "CREATE INDEX IF NOT EXISTS idx_vendors_cat       ON vendors(category)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_cat_lower ON vendors(lower(category))",
        "CREATE INDEX IF NOT EXISTS idx_vendors_svc_lower ON vendors(lower(service))",
    ]

    def _split_sql(s: str) -> list[str]:
        # Conservative split on ';', remove empties; keep order
        parts = [p.strip() for p in s.replace("\r", "").split(";")]
        return [p for p in parts if p]

    flattened: list[str] = []
    for s in stmts:
        flattened.extend(_split_sql(s))

    from sqlalchemy import exc as sa_exc

    with engine.begin() as conn:
        for stmt in flattened:
            if show_debug:
                st.write(":wrench: **DDL** →", f"`{stmt}`")
            try:
                conn.exec_driver_sql(stmt)
            except sa_exc.OperationalError as e:
                msg = str(e).lower()
                if ("already exists" in msg) or ("duplicate" in msg):
                    if show_debug:
                        st.write(":information_source: skipped benign error:", str(e))
                    continue
                raise

        # --- migration: add computed_keywords if missing ---
        cols = conn.execute(sql_text("PRAGMA table_info(vendors)")).fetchall()
        colnames = {r[1] for r in cols}
        if "computed_keywords" not in colnames:
            try:
                conn.exec_driver_sql("ALTER TABLE vendors ADD COLUMN computed_keywords TEXT")
            except Exception:
                pass
        # index for computed keywords
        try:
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_vendors_ckw ON vendors(computed_keywords)")
        except Exception:
            pass


# -----------------------------
# Computed keywords generator
# -----------------------------
# Expandable synonym map by service (lowercase keys)
SVC_SYNONYMS: Dict[str, List[str]] = {
    "plumbing": ["plumber", "pipe repair", "leak", "water heater", "drain", "sewer", "toilet", "faucet"],
    "roofing": ["roofer", "shingles", "leak repair", "re-roof", "hail damage", "gutters"],
    "hvac, air conditioning, heating": ["hvac", "ac repair", "air conditioner", "furnace", "heater", "thermostat", "duct"],
    "pest control": ["exterminator", "bug spray", "ants", "roaches", "termites", "spiders", "fleas", "ticks", "wasps"],
    "window coverings": ["blinds", "shades", "shutters", "plantation shutters", "drapes", "curtains"],
    "garage doors": ["overhead door", "spring repair", "opener", "door install", "panel", "tracks"],
    "flooring": ["tile", "hardwood", "laminate", "vinyl plank", "carpet", "installation", "refinish"],
    "landscaping": ["yard", "lawn", "mulch", "sod", "hedge", "sprinkler", "irrigation", "tree trimming"],
    "tree removal": ["tree service", "stump grinding", "arborist", "tree trimming"],
    "fencing": ["privacy fence", "wood fence", "wrought iron", "gate", "repair", "install"],
    "carpet cleaning": ["steam clean", "stain removal", "odor", "upholstery cleaning", "pet stains"],
    "auto repair": ["mechanic", "oil change", "brakes", "tires", "alignment", "check engine"],
    "optometry / eye exam": ["eye doctor", "optometrist", "glasses", "contact lenses", "vision exam"],
    "ophthalmologist": ["eye surgeon", "cataract", "lasik", "retina"],
    "dental": ["dentist", "teeth cleaning", "fillings", "crown", "root canal", "whitening"],
    "dermatology": ["dermatologist", "skin cancer", "mohs", "acne", "rash", "psoriasis", "moles"],
    "orthopedics": ["orthopedic", "bone", "joint", "knee", "hip", "shoulder", "sports medicine"],
    "chiropractic": ["chiropractor", "back pain", "adjustment", "spine", "neck pain"],
    "aud iology": ["hearing test", "hearing aids", "tinnitus"],  # tolerate odd spacing seen in imports
    "gutter cleaning": ["gutter wash", "downspout", "roof wash", "soft wash"],
    "window washing": ["window cleaning", "glass cleaning", "squeegee"],
    "deck staining": ["deck refinish", "sealant", "wood stain", "fence stain"],
    "address tiles": ["house numbers", "address plaque", "numbers tile"],
}

# Aliases to normalize common variants → canonical service key used above
SERVICE_ALIASES: Dict[str, str] = {
    "hvac": "hvac, air conditioning, heating",
    "air conditioning": "hvac, air conditioning, heating",
    "heating": "hvac, air conditioning, heating",
    "optometry": "optometry / eye exam",
    "eye exam": "optometry / eye exam",
    "ophthalmology": "ophthalmologist",
    "dentistry": "dental",
    "tree trim": "tree removal",
    "tree services": "tree removal",
    "pest control": "pest control",
    "window coverings": "window coverings",
    "garage door": "garage doors",
    "flooring, walls": "flooring",
    "carpet cleaning": "carpet cleaning",
    "window washing": "window washing",
    "deck staining": "deck staining",
    "address tiles": "address tiles",
    "auto repair": "auto repair",
    "landscaping & outdoor services": "landscaping",
}


def _norm_txt(s: Optional[str]) -> str:
    return (s or "").strip()


def _lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _service_key(raw_service: Optional[str]) -> str:
    s = _lower(raw_service)
    if not s:
        return ""
    return SERVICE_ALIASES.get(s, s)


def generate_computed_keywords(category: Optional[str], service: Optional[str], business_name: Optional[str]) -> str:
    """
    Produce a space-separated keyword string.
    Rule: if service is blank → return "" (no computed keywords).
    Otherwise: service-centric terms + generic add-ons + a few from business_name.
    """
    cat = _lower(category)
    svc_raw = _lower(service)
    if not svc_raw:
        return ""

    svc = _service_key(svc_raw)

    # Base terms: include raw service text
    terms: List[str] = []

    # Include canonical service tokens
    terms.append(svc)

    # Service synonyms
    svc_syn = SVC_SYNONYMS.get(svc, [])
    terms.extend(svc_syn)

    # Category hints (very light add)
    cat_hints = {
        "landscaping & outdoor services": ["outdoor", "yard", "garden"],
        "home repair & trades": ["contractor", "repair", "install"],
        "tree services": ["tree", "arborist"],
        "plumbing": ["plumbing"],
    }
    if cat in cat_hints:
        terms.extend(cat_hints[cat])

    # Business name cues (very conservative, only obvious functional nouns)
    bn = _lower(business_name)
    for token in ("curb", "border", "collision", "shutter", "grout", "marble", "alarm", "roof", "floor", "window", "deck", "patio", "sprinkler", "irrigation", "upholstery", "vet", "dental", "eye", "orthopedic", "endodontic", "granite"):
        if token in bn:
            terms.append(token)

    # De-dup while preserving order
    seen = set()
    out: List[str] = []
    for t in terms:
        t = t.strip()
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)

    return " ".join(out)


# -----------------------------
# Data access helpers (DECLARED BEFORE UI)
# -----------------------------
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
        "computed_keywords",  # <-- ensure present in DF
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


def _ensure_ref_value(engine: Engine, table: str, value: str) -> None:
    """Insert a category/service name if it's new (no-op if exists)."""
    v = (value or "").strip()
    if not v:
        return
    _exec_with_retry(engine, f"INSERT OR IGNORE INTO {table}(name) VALUES(:n)", {"n": v})


# -----------------------------
# Page config & CSS
# -----------------------------
PAGE_TITLE = _resolve_str("page_title", "Vendors Admin") or "Vendors Admin"
SIDEBAR_STATE = _resolve_str("sidebar_state", "expanded") or "expanded"
st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

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
# Admin sign-in gate (deterministic toggle)
# -----------------------------
DISABLE_ADMIN_PASSWORD_DEFAULT = True      # True = bypass, False = require password
ADMIN_PASSWORD_DEFAULT = "admin"

DISABLE_LOGIN = _resolve_bool("DISABLE_ADMIN_PASSWORD", DISABLE_ADMIN_PASSWORD_DEFAULT)
ADMIN_PASSWORD = (_resolve_str("ADMIN_PASSWORD", ADMIN_PASSWORD_DEFAULT) or "").strip()

if DISABLE_LOGIN:
    pass
else:
    if not ADMIN_PASSWORD:
        st.error("ADMIN_PASSWORD is not set (Secrets/Env).")
        st.stop()
    if "auth_ok" not in st.session_state:
        st.session_state["auth_ok"] = False
    if not st.session_state["auth_ok"]:
        st.subheader("Admin sign-in")
        pw = st.text_input("Password", type="password", key="admin_pw")
        if st.button("Sign in"):
            if _ct_equals((pw or "").strip(), ADMIN_PASSWORD):
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()


# -----------------------------
# UI (engine + schema init occurs before UI uses helpers)
# -----------------------------
engine, engine_info = build_engine()
ensure_schema(engine, show_debug=_resolve_bool("SCHEMA_DEBUG", False))

# Apply WAL PRAGMAs for local SQLite (not libsql driver)
try:
    if not engine_info.get("using_remote", False) and engine_info.get("driver", "") != "libsql":
        with engine.begin() as _conn:
            _conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
            _conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
except Exception:
    pass


# -----------------------------
# Startup backfill hook (idempotent)
# -----------------------------
def _backfill_computed_keywords(engine: Engine) -> Dict[str, int]:
    """
    Compute computed_keywords for rows where service is present and computed_keywords is NULL/blank.
    """
    updated = 0
    skipped_service_blank = 0

    with engine.begin() as conn:
        rows = conn.execute(
            sql_text(
                """
                SELECT id, category, service, business_name, computed_keywords
                  FROM vendors
                 WHERE (computed_keywords IS NULL OR TRIM(computed_keywords) = '')
                """
            )
        ).mappings().all()

        for r in rows:
            svc = _norm_txt(r["service"])
            if not svc:
                skipped_service_blank += 1
                continue
            ckw = generate_computed_keywords(r["category"], r["service"], r["business_name"])
            conn.execute(
                sql_text("UPDATE vendors SET computed_keywords=:ckw WHERE id=:id"),
                {"ckw": ckw, "id": int(r["id"])},
            )
            updated += 1

    return {"updated": updated, "skipped_service_blank": skipped_service_blank}


# fire once on load (cheap)
try:
    _ = _backfill_computed_keywords(engine)
except Exception:
    pass


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
            for c in [
                "business_name",
                "category",
                "service",
                "contact_name",
                "phone",
                "address",
                "website",
                "notes",
                "keywords",
                "computed_keywords",  # include computed keywords in search
            ]
        ]
        df["_blob"] = pd.concat(parts, axis=1).agg(" ".join, axis=1).str.lower()

    # --- Search input (Enter) ---
    left, right = st.columns([1, 3])
    with left:
        q = st.text_input(
            "Search",
            placeholder="Search providers… (press Enter)",
            label_visibility="collapsed",
            key="q",
        )

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
        "computed_keywords",  # show in admin browse
    ]

    vdf = filtered[view_cols].rename(columns={"phone_fmt": "phone"})

    st.dataframe(
        vdf,
        use_container_width=True,
        hide_index=True,
        column_config={
            "business_name": st.column_config.TextColumn("Provider"),
            "website": st.column_config.LinkColumn("website"),
            "notes": st.column_config.TextColumn(width=420),
            "keywords": st.column_config.TextColumn(width=300),
            "computed_keywords": st.column_config.TextColumn(width=900),
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
    # Form state helpers
    ADD_FORM_KEYS = [
        "add_business_name",
        "add_category",
        "add_service",
        "add_contact_name",
        "add_phone",
        "add_address",
        "add_website",
        "add_notes",
        "add_keywords",
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
        if st.session_state.get("_pending_add_reset"):
            for k in ADD_FORM_KEYS:
                st.session_state[k] = ""
            st.session_state["_pending_add_reset"] = False
            st.session_state["add_form_version"] += 1

    def _queue_add_form_reset():
        st.session_state["_pending_add_reset"] = True

    # init/apply
    _init_add_form_defaults()
    _apply_add_reset_if_needed()

    cats = list_names(engine, "categories")
    servs = list_names(engine, "services")

    add_form_key = f"add_vendor_form_{st.session_state['add_form_version']}"
    with st.form(add_form_key, clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Provider *", key="add_business_name")

            _add_cat_options = [""] + (cats or [])
            if (st.session_state.get("add_category") or "") not in _add_cat_options:
                st.session_state["add_category"] = ""
            st.selectbox("Category *", options=_add_cat_options, key="add_category", placeholder="Select category")

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

    def _nonce(name: str) -> str:
        return st.session_state.get(f"{name}_nonce")

    def _nonce_rotate(name: str) -> None:
        st.session_state[f"{name}_nonce"] = uuid.uuid4().hex

    if submitted:
        add_nonce = _nonce("add")
        if st.session_state.get("add_last_done") == add_nonce:
            st.info("Add already processed.")
            st.stop()

        business_name = (st.session_state["add_business_name"] or "").strip()
        category = (st.session_state["add_category"] or "").strip()
        service = (st.session_state["add_service"] or "").strip()
        contact_name = (st.session_state["add_contact_name"] or "").strip()
        phone_norm = _normalize_phone(st.session_state["add_phone"])
        address = (st.session_state["add_address"] or "").strip()
        website = _sanitize_url(st.session_state["add_website"])
        notes = (st.session_state["add_notes"] or "").strip()
        keywords = (st.session_state["add_keywords"] or "").strip()

        # Minimal validation
        if phone_norm and len(phone_norm) != 10:
            st.error("Phone must be 10 digits or blank.")
        elif not business_name or not category:
            st.error("Business Name and Category are required.")
        else:
            try:
                # Ensure ref values exist if new
                _ensure_ref_value(engine, "categories", category)
                if service:
                    _ensure_ref_value(engine, "services", service)

                ckw = generate_computed_keywords(category, service, business_name)

                now = datetime.utcnow().isoformat(timespec="seconds")
                _exec_with_retry(
                    engine,
                    """
                    INSERT INTO vendors(category, service, business_name, contact_name, phone, address,
                                        website, notes, keywords, computed_keywords, created_at, updated_at, updated_by)
                    VALUES(:category, NULLIF(:service, ''), :business_name, :contact_name, :phone, :address,
                           :website, :notes, :keywords, :ckw, :now, :now, :user)
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
                        "ckw": ckw,
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

    # ----- Edit/Delete form state helpers -----
    EDIT_FORM_KEYS = [
        "edit_vendor_id",
        "edit_business_name",
        "edit_category",
        "edit_service",
        "edit_contact_name",
        "edit_phone",
        "edit_address",
        "edit_website",
        "edit_notes",
        "edit_keywords",
        "edit_row_updated_at",
        "edit_last_loaded_id",
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

    df_all = load_df(engine)

    if df_all.empty:
        st.info("No vendors yet. Use 'Add Vendor' above to create your first record.")
    else:
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
                st.session_state.update(
                    {
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
                    }
                )

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
                    # don't force-blank; keep user's current text. We'll allow free text via text_input below.
                    pass
                # swap selectbox for text_input to permit new entries directly
                st.text_input("Category *", key="edit_category", placeholder="Type or pick an existing")

                _edit_svc_options = [""] + (servs or [])
                if (st.session_state.get("edit_service") or "") not in _edit_svc_options:
                    pass
                st.text_input("Service (optional)", key="edit_service", placeholder="Type or pick an existing")

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
                bn = (st.session_state["edit_business_name"] or "").strip()
                cat = (st.session_state["edit_category"] or "").strip()
                svc = (st.session_state["edit_service"] or "").strip()
                phone_norm = _normalize_phone(st.session_state["edit_phone"])
                if phone_norm and len(phone_norm) != 10:
                    st.error("Phone must be 10 digits or blank.")
                elif not bn or not cat:
                    st.error("Business Name and Category are required.")
                else:
                    try:
                        # Ensure ref values exist (free-text allowed)
                        _ensure_ref_value(engine, "categories", cat)
                        if svc:
                            _ensure_ref_value(engine, "services", svc)

                        ckw = generate_computed_keywords(cat, svc, bn)

                        prev_updated = st.session_state.get("edit_row_updated_at") or ""
                        now = datetime.utcnow().isoformat(timespec="seconds")
                        res = _exec_with_retry(
                            engine,
                            """
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
                                   updated_at=:now,
                                   updated_by=:user
                             WHERE id=:id AND (updated_at=:prev_updated OR :prev_updated='')
                            """,
                            {
                                "category": cat,
                                "service": svc,
                                "business_name": bn,
                                "contact_name": (st.session_state["edit_contact_name"] or "").strip(),
                                "phone": phone_norm,
                                "address": (st.session_state["edit_address"] or "").strip(),
                                "website": _sanitize_url(st.session_state["edit_website"]),
                                "notes": (st.session_state["edit_notes"] or "").strip(),
                                "keywords": (st.session_state["edit_keywords"] or "").strip(),
                                "ckw": ckw,
                                "now": now,
                                "user": os.getenv("USER", "admin"),
                                "id": int(vid),
                                "prev_updated": prev_updated,
                            },
                        )
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
            options=["— Select —"] + [_fmt_vendor(i) for i in ids],
            key="delete_provider_label",
        )
        if sel_label_del != "— Select —":
            rev = {_fmt_vendor(i): i for i in ids}
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
                    res = _exec_with_retry(
                        engine,
                        """
                        DELETE FROM vendors
                         WHERE id=:id AND (updated_at=:prev_updated OR :prev_updated='')
                        """,
                        {"id": int(vid), "prev_updated": prev_updated},
                    )
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

    # Category state helpers
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
            old = st.selectbox("Current", options=cat_opts, key="cat_old")
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
            tgt = st.selectbox("Category to delete", options=cat_opts, key="cat_del")
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
            old = st.selectbox("Current", options=svc_opts, key="svc_old")  # no index default
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
                    repl = st.selectbox("Reassign vendors to…", options=repl_options, key="svc_reassign_to")
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

    query = "SELECT * FROM vendors ORDER BY lower(business_name)"
    with engine.begin() as conn:
        full = pd.read_sql(sql_text(query), conn)

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

    # NOTE: CSV Restore helpers (_prepare_csv_for_append / _execute_append_only)
    # were part of your existing app. They continue to work; computed_keywords will
    # be accepted if present in the CSV but is not required.

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

                # --- categories table ---
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

                # --- services table ---
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
