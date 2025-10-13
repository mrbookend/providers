# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import hmac
import time
import uuid
from datetime import datetime
from typing import List, Tuple, Dict, Iterable, Any, Optional

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


def _resolve_bool(name: str, code_default: bool) -> bool:
    v = _get_secret(name, None)
    return _as_bool(v, default=code_default)


def _resolve_str(name: str, code_default: str | None) -> str | None:
    v = _get_secret(name, None)
    return v if v is not None else code_default


def _ct_equals(a: str, b: str) -> bool:
    return hmac.compare_digest((a or ""), (b or ""))


# -----------------------------
# Hrana/libSQL transient error retry
# -----------------------------
def _is_hrana_stale_stream_error(err: Exception) -> bool:
    s = str(err).lower()
    return ("hrana" in s and "404" in s and "stream not found" in s) or ("stream not found" in s)


def _exec_with_retry(engine: Engine, sql: str, params: Dict | None = None, *, tries: int = 2):
    attempt = 0
    while True:
        attempt += 1
        try:
            with engine.begin() as conn:
                return conn.execute(sql_text(sql), params or {})
        except Exception as e:
            if attempt < tries and _is_hrana_stale_stream_error(e):
                try:
                    engine.dispose()
                except Exception:
                    pass
                time.sleep(0.2)
                continue
            raise


def _fetch_with_retry(engine: Engine, sql: str, params: Dict | None = None, *, tries: int = 2) -> pd.DataFrame:
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
    info: Dict = {}

    url = (_resolve_str("TURSO_DATABASE_URL", "") or "").strip()
    token = (_resolve_str("TURSO_AUTH_TOKEN", "") or "").strip()
    embedded_path = os.path.abspath(_resolve_str("EMBEDDED_DB_PATH", "vendors-embedded.db") or "vendors-embedded.db")

    if not url:
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

    try:
        raw = url
        if raw.startswith("sqlite+libsql://"):
            host = raw.split("://", 1)[1].split("?", 1)[0]
            sync_url = f"libsql://{host}"
        else:
            sync_url = raw.split("?", 1)[0]

        eng = create_engine(
            f"sqlite+libsql:///{embedded_path}",
            connect_args={"auth_token": token, "sync_url": sync_url},
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
    Defensive schema init:
      1) Create tables if missing
      2) Add 'computed_keywords' column only if not present
      3) Create indexes (IF NOT EXISTS)
    """
    from sqlalchemy import exc as sa_exc

    def _run(conn: Engine, stmt: str):
        if show_debug:
            st.write(":wrench: **DDL** →", f"`{stmt}`")
        try:
            conn.exec_driver_sql(stmt)
        except sa_exc.OperationalError as e:
            msg = str(e).lower()
            # Ignore benign dup errors across sqlite/libsql variants
            if ("already exists" in msg) or ("duplicate" in msg):
                return
            raise

    with engine.begin() as conn:
        # 1) Tables
        _run(conn, """
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
        )""")
        _run(conn, """
        CREATE TABLE IF NOT EXISTS categories (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT UNIQUE
        )""")
        _run(conn, """
        CREATE TABLE IF NOT EXISTS services (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT UNIQUE
        )""")

        # 2) Conditionally add computed_keywords (check PRAGMA first)
        cols = conn.exec_driver_sql("PRAGMA table_info(vendors)").fetchall()
        colnames = {c[1] for c in cols}
        if "computed_keywords" not in colnames:
            try:
                _run(conn, "ALTER TABLE vendors ADD COLUMN computed_keywords TEXT")
            except Exception:
                pass

        # 3) Indexes
        _run(conn, "CREATE INDEX IF NOT EXISTS idx_vendors_cat ON vendors(category)")
        _run(conn, "CREATE INDEX IF NOT EXISTS idx_vendors_bus ON vendors(business_name)")
        _run(conn, "CREATE INDEX IF NOT EXISTS idx_vendors_kw  ON vendors(keywords)")
        _run(conn, "CREATE INDEX IF NOT EXISTS idx_vendors_bus_lower ON vendors(lower(business_name))")
        _run(conn, "CREATE INDEX IF NOT EXISTS idx_vendors_cat_lower ON vendors(lower(category))")
        _run(conn, "CREATE INDEX IF NOT EXISTS idx_vendors_svc_lower ON vendors(lower(service))")
        _run(conn, "CREATE INDEX IF NOT EXISTS idx_vendors_phone ON vendors(phone)")
        _run(conn, "CREATE INDEX IF NOT EXISTS idx_vendors_ckw ON vendors(computed_keywords)")


# -----------------------------
# Computed keywords
# -----------------------------
SERVICE_ALIASES: Dict[str, List[str]] = {
    "Plumbing": ["plumber", "leak repair", "water heater", "drain", "sewer", "pipe", "toilet", "faucet"],
    "Roofing": ["roofer", "roof repair", "hail damage", "shingles", "leak"],
    "HVAC, Air Conditioning, Heating": ["hvac", "ac", "air conditioner", "furnace", "heater", "duct", "thermostat", "cooling", "heating"],
    "Handyman": ["repairs", "odd jobs", "honey-do", "maintenance"],
    "Window Coverings": ["blinds", "shutters", "shades", "plantation shutters", "drapes"],
    "Pest Control": ["exterminator", "bugs", "insects", "ants", "spiders", "termites", "fleas", "ticks", "roach", "mosquito", "spraying"],
    "Tree Removal": ["tree service", "stump grinding", "arborist", "tree cutting", "tree trimming"],
    "Tree Trim": ["tree service", "pruning", "arborist", "branch removal"],
    "Garage Doors": ["overhead door", "door opener", "springs", "garage repair", "panel"],
    "Carpet Cleaning": ["steam clean", "stain removal", "upholstery cleaning"],
    "Window Washing": ["window cleaning", "glass cleaning"],
    "Flooring": ["hardwood", "tile", "vinyl plank", "laminate", "carpet install", "refinish"],
    "Decks, Gazebo, Arbor, Pergola": ["patio cover", "pergola", "gazebo", "deck build", "outdoor living"],
    "Concrete Resurfacing": ["stamped concrete", "overlay", "cool deck", "resurface"],
    "Irrigation": ["sprinkler repair", "sprinklers", "backflow", "irrigation system"],
    "Lawn Care": ["mowing", "edging", "fertilizer", "weed control", "aeration"],
    "Artificial Turf": ["synthetic grass", "astroturf", "fake grass"],
    "Fencing": ["fence install", "privacy fence", "gate", "fence repair"],
    "Grout Clean": ["tile cleaning", "grout sealing", "grout repair"],
    "Softener Service, Sales": ["water softener", "reverse osmosis", "ro system", "salt delivery"],
    "Auto Repair": ["mechanic", "brakes", "oil change", "alignment", "diagnostics"],
    "Dental": ["dentist", "teeth", "cleaning", "fillings", "crown"],
    "Chiropractic": ["back pain", "adjustment", "spine"],
    "Dermatology": ["skin doctor", "mohs", "acne", "biopsy"],
    "Optometry / Eye Exam": ["eye doctor", "optometrist", "glasses", "contacts", "eye exam"],
    "Orthopedics": ["orthopedic", "joint", "knee", "shoulder", "hip", "sports medicine"],
    "Primary Care": ["family doctor", "pcp", "checkup", "internal medicine"],
    "Audiology": ["hearing test", "hearing aids", "tinnitus"],
}
SVC_SYNONYMS = {k.lower(): v for k, v in SERVICE_ALIASES.items()}


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


def _computed_keywords_for(category: str, service: str, business_name: str = "") -> str:
    cat = (category or "").strip()
    svc = (service or "").strip()
    name = (business_name or "").strip()
    if not svc:
        return ""

    tokens: List[str] = []
    for raw in (cat, svc, name):
        for t in re.split(r"[^\w]+", raw):
            t = t.strip().lower()
            if len(t) >= 2:
                tokens.append(t)

    syns = SVC_SYNONYMS.get(svc.lower(), [])
    for s in syns:
        s = s.strip().lower()
        if s:
            tokens.extend(re.split(r"[^\w]+", s))

    seen = set()
    out: List[str] = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)

    return " ".join(out)


def _recompute_missing_computed_keywords(engine: Engine) -> Dict[str, int]:
    now = datetime.utcnow().isoformat(timespec="seconds")
    updated = 0
    checked = 0
    with engine.begin() as conn:
        rows = conn.execute(
            sql_text(
                """
                SELECT id, category, service, business_name
                  FROM vendors
                 WHERE (computed_keywords IS NULL OR computed_keywords = '')
                   AND service IS NOT NULL AND TRIM(service) <> ''
                """
            )
        ).mappings().all()

        for r in rows:
            checked += 1
            ck = _computed_keywords_for(r["category"], r["service"], r["business_name"])
            if ck:
                conn.execute(
                    sql_text(
                        """
                        UPDATE vendors
                           SET computed_keywords=:ck,
                               updated_at = COALESCE(updated_at, :now)
                         WHERE id=:id
                        """
                    ),
                    {"ck": ck, "id": r["id"], "now": now},
                )
                updated += 1
    return {"checked": checked, "updated": updated}


def recompute_keywords_all(engine: Engine) -> Dict[str, int]:
    """Rebuild computed_keywords for ALL rows with a non-empty service using synonym logic."""
    updated = 0
    checked = 0
    now = datetime.utcnow().isoformat(timespec="seconds")
    with engine.begin() as conn:
        rows = conn.execute(
            sql_text(
                """
                SELECT id, category, service, business_name
                  FROM vendors
                 WHERE service IS NOT NULL AND TRIM(service) <> ''
                """
            )
        ).mappings().all()
        for r in rows:
            checked += 1
            ck = _computed_keywords_for(r["category"], r["service"], r["business_name"])
            conn.execute(
                sql_text(
                    """
                    UPDATE vendors
                       SET computed_keywords=:ck,
                           updated_at = COALESCE(updated_at, :now)
                     WHERE id=:id
                    """
                ),
                {"ck": ck, "id": r["id"], "now": now},
            )
            updated += 1
    return {"checked": checked, "updated": updated}


# -----------------------------
# Quick Probes (read-only health checks)
# -----------------------------
def _run_quick_probes(engine: Engine) -> dict[str, Any]:
    results: dict[str, Any] = {}

    with engine.connect() as conn:
        results["counts"] = {
            "vendors": conn.execute(sql_text("SELECT COUNT(*) FROM vendors")).scalar_one(),
            "categories": conn.execute(sql_text("SELECT COUNT(*) FROM categories")).scalar_one(),
            "services": conn.execute(sql_text("SELECT COUNT(*) FROM services")).scalar_one(),
        }

        try:
            results["integrity_check"] = conn.execute(sql_text("PRAGMA integrity_check;")).scalar_one()
        except Exception as e:
            results["integrity_check"] = f"unavailable: {e}"

        try:
            idx_rows = conn.execute(sql_text("PRAGMA index_list('vendors');")).fetchall()
            results["vendors_indexes"] = [
                {"seq": r[0], "name": r[1], "unique": bool(r[2])} for r in idx_rows
            ]
        except Exception as e:
            results["vendors_indexes"] = f"unavailable: {e}"

        ts = conn.execute(sql_text("""
            SELECT
              SUM(CASE WHEN created_at IS NULL OR created_at='' THEN 1 ELSE 0 END) AS created_at_nulls,
              SUM(CASE WHEN updated_at IS NULL OR updated_at='' THEN 1 ELSE 0 END) AS updated_at_nulls
            FROM vendors
        """)).one()
        results["timestamp_nulls"] = {"created_at": ts[0] or 0, "updated_at": ts[1] or 0}

        ckw = conn.execute(sql_text("""
            SELECT
              SUM(CASE WHEN computed_keywords IS NULL OR TRIM(computed_keywords) = '' THEN 1 ELSE 0 END) AS missing,
              COUNT(*) AS total
            FROM vendors
        """)).one()
        results["computed_keywords"] = {"missing": ckw[0] or 0, "total": ckw[1] or 0}

        dupes = conn.execute(sql_text("""
            SELECT LOWER(TRIM(business_name)) AS b, LOWER(TRIM(category)) AS c,
                   LOWER(TRIM(COALESCE(service,''))) AS s, COUNT(*) AS n
            FROM vendors
            GROUP BY b, c, s
            HAVING COUNT(*) > 1
            ORDER BY n DESC, b, c, s
            LIMIT 100
        """)).fetchall()
        results["possible_duplicates_count"] = len(dupes)
        results["_dupes_rows"] = [dict(zip(("b","c","s","n"), r)) for r in dupes]

        bad_phones = conn.execute(sql_text("""
            SELECT id, business_name, phone
            FROM vendors
            WHERE phone IS NOT NULL AND TRIM(phone) <> ''
              AND LENGTH(REPLACE(REPLACE(REPLACE(REPLACE(phone,'-',''),'(',''),')',''),' ',''))
                    <> 10
            LIMIT 200
        """)).fetchall()
        results["bad_phones_count"] = len(bad_phones)
        results["_bad_phones_rows"] = [dict(zip(("id","business_name","phone"), r)) for r in bad_phones]

        bad_sites = conn.execute(sql_text("""
            SELECT id, business_name, website
            FROM vendors
            WHERE website IS NOT NULL AND TRIM(website) <> ''
              AND LOWER(website) NOT LIKE 'http://%%'
              AND LOWER(website) NOT LIKE 'https://%%'
            LIMIT 200
        """)).fetchall()
        results["bad_websites_count"] = len(bad_sites)
        results["_bad_websites_rows"] = [dict(zip(("id","business_name","website"), r)) for r in bad_sites]

        orphan_cat = conn.execute(sql_text("""
            SELECT v.id, v.business_name, v.category
            FROM vendors v
            LEFT JOIN categories c ON LOWER(TRIM(v.category)) = LOWER(TRIM(c.name))
            WHERE v.category IS NOT NULL AND TRIM(v.category) <> ''
              AND c.id IS NULL
            LIMIT 200
        """)).fetchall()
        results["orphan_categories_count"] = len(orphan_cat)
        results["_orphan_categories_rows"] = [dict(zip(("id","business_name","category"), r)) for r in orphan_cat]

        orphan_svc = conn.execute(sql_text("""
            SELECT v.id, v.business_name, v.service
            FROM vendors v
            LEFT JOIN services s ON LOWER(TRIM(v.service)) = LOWER(TRIM(s.name))
            WHERE v.service IS NOT NULL AND TRIM(v.service) <> ''
              AND s.id IS NULL
            LIMIT 200
        """)).fetchall()
        results["orphan_services_count"] = len(orphan_svc)
        results["_orphan_services_rows"] = [dict(zip(("id","business_name","service"), r)) for r in orphan_svc]

        unused_cat = conn.execute(sql_text("""
            SELECT c.name
            FROM categories c
            LEFT JOIN vendors v ON LOWER(TRIM(v.category)) = LOWER(TRIM(c.name))
            WHERE v.id IS NULL
            ORDER BY c.name
            LIMIT 200
        """)).fetchall()
        results["unused_categories_count"] = len(unused_cat)
        results["_unused_categories_rows"] = [dict(zip(("name",), r)) for r in unused_cat]

        unused_svc = conn.execute(sql_text("""
            SELECT s.name
            FROM services s
            LEFT JOIN vendors v ON LOWER(TRIM(v.service)) = LOWER(TRIM(s.name))
            WHERE v.id IS NULL
            ORDER BY s.name
            LIMIT 200
        """)).fetchall()
        results["unused_services_count"] = len(unused_svc)
        results["_unused_services_rows"] = [dict(zip(("name",), r)) for r in unused_svc]

    return results


def _render_quick_probes_ui(engine: Engine) -> None:
    if st.button("Run Quick Probes", type="primary"):
        data = _run_quick_probes(engine)
        st.json({k: v for k, v in data.items() if not k.startswith("_")})
        for key, label in [
            ("_dupes_rows", "Possible duplicates"),
            ("_bad_phones_rows", "Bad phone formats"),
            ("_bad_websites_rows", "Bad website formats"),
            ("_orphan_categories_rows", "Orphan categories (not in library)"),
            ("_orphan_services_rows", "Orphan services (not in library)"),
            ("_unused_categories_rows", "Unused categories"),
            ("_unused_services_rows", "Unused services"),
        ]:
            rows = data.get(key, [])
            if isinstance(rows, list) and rows:
                st.write(f"**{label}** ({len(rows)})")
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# -----------------------------
# Data access helpers
# -----------------------------
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
        "computed_keywords",
    ]:
        if col not in df.columns:
            df[col] = ""

    df["phone_fmt"] = df["phone"].apply(_format_phone)
    return df


def list_names(engine: Engine, table: str) -> list[str]:
    with engine.begin() as conn:
        rows = conn.execute(sql_text(f"SELECT name FROM {table} ORDER BY lower(name)")).fetchall()
    return [r[0] for r in rows]


def list_names_normalized(engine: Engine, table: str) -> list[str]:
    raw = list_names(engine, table)
    return [str(x).strip() for x in raw if (x or "").strip()]


def usage_count(engine: Engine, col: str, name: str) -> int:
    with engine.begin() as conn:
        cnt = conn.execute(sql_text(f"SELECT COUNT(*) FROM vendors WHERE {col} = :n"), {"n": name}).scalar()
    return int(cnt or 0)


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
      div[data-testid="stDataFrame"] table {{
        white-space: nowrap;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Admin sign-in gate
# -----------------------------
DISABLE_ADMIN_PASSWORD_DEFAULT = True
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
# UI (engine + schema init)
# -----------------------------
engine, engine_info = build_engine()
ensure_schema(engine, show_debug=_resolve_bool("SCHEMA_DEBUG", False))

try:
    if not engine_info.get("using_remote", False) and engine_info.get("driver", "") != "libsql":
        with engine.begin() as _conn:
            _conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
            _conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
except Exception:
    pass

try:
    stats = _recompute_missing_computed_keywords(engine)
    if (stats.get("updated", 0) or 0) > 0 and _resolve_bool("SHOW_COMPUTED_BACKFILL", False):
        st.info(f"Backfilled computed_keywords on startup: {stats}")
except Exception as _e:
    if _resolve_bool("SHOW_COMPUTED_BACKFILL", False):
        st.warning(f"Backfill skipped: {_e}")


# =============================
# Tabs
# =============================
_tabs = st.tabs(
    [
        "Browse Providers",          # ← renamed
        "Add / Edit / Delete Provider",
        "Category Admin",
        "Service Admin",
        "System Tools / Help",       # unified maintenance/help/debug/probes/import/export
    ]
)

# ---------- Browse Providers
with _tabs[0]:
    df = load_df(engine)

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
                "computed_keywords",
            ]
        ]
        df["_blob"] = pd.concat(parts, axis=1).agg(" ".join, axis=1).str.lower()

    left, right = st.columns([1, 3])
    with left:
        st.text_input("Search", placeholder="Search providers… (press Enter)", label_visibility="collapsed", key="q")

    qq = (st.session_state.get("q") or "").strip().lower()
    filtered = df[df["_blob"].str.contains(qq, regex=False, na=False)] if qq else df

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
        "__spacer",
    ]

    if "__spacer" not in filtered.columns:
        filtered["__spacer"] = ""

    vdf = filtered[view_cols].rename(columns={"phone_fmt": "phone"})

    st.dataframe(
        vdf,
        use_container_width=True,
        hide_index=True,
        column_config={
            "business_name": st.column_config.TextColumn("Provider", width=220),
            "category": st.column_config.TextColumn(width=180),
            "service": st.column_config.TextColumn(width=200),
            "phone": st.column_config.TextColumn(width=120),
            "address": st.column_config.TextColumn(width=260),
            "website": st.column_config.LinkColumn("website", width=200),
            "notes": st.column_config.TextColumn(width=260),
            "keywords": st.column_config.TextColumn(width=220),
            "computed_keywords": st.column_config.TextColumn(width=500),
            "__spacer": st.column_config.TextColumn(label="", width=120),
        },
    )

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    vdf_no_spacer = vdf.drop(columns="__spacer")
    st.download_button(
        "Download filtered view (CSV)",
        data=vdf_no_spacer.to_csv(index=False).encode("utf-8"),
        file_name=f"providers_{ts}.csv",
        mime="text/csv",
    )

# ---------- Add / Edit / Delete Provider  (side-by-side Add & Edit; Delete at bottom)
with _tabs[1]:
    # NOTE: Removed st.header("Add / Edit Provider") per request.

    # Shared state
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

    # Init
    _init_add_form_defaults()
    _apply_add_reset_if_needed()
    _init_edit_form_defaults()
    _apply_edit_reset_if_needed()
    _init_delete_form_defaults()
    _apply_delete_reset_if_needed()

    cats = list_names_normalized(engine, "categories")
    servs = list_names_normalized(engine, "services")

    df_all = load_df(engine)
    ids = df_all["id"].astype(int).tolist() if not df_all.empty else []
    id_to_row = {int(r["id"]): r for _, r in df_all.iterrows()} if not df_all.empty else {}

    def _fmt_provider(i: int | None) -> str:
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

    # Two columns: Add (left), Edit (right)
    col_left, col_right = st.columns(2)

    # LEFT — Add Provider
    with col_left:
        st.subheader("Add Provider")
        add_form_key = f"add_provider_form_{st.session_state['add_form_version']}"
        with st.form(add_form_key, clear_on_submit=False):
            # Row 0: spacer to align with the Edit selectbox height
            st.markdown("<div style='height: 70px;'></div>", unsafe_allow_html=True)

            # Row 1..: inputs laid out in two columns to mirror Edit
            c1, c2 = st.columns(2)
            with c1:
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
            with c2:
                st.text_area("Address", height=80, key="add_address")
                st.text_input("Website (https://…)", key="add_website")
                st.text_area("Notes", height=100, key="add_notes")
                st.text_input("Keywords (comma separated)", key="add_keywords")

            add_clicked = st.form_submit_button("Add Provider")

        def _nonce(name: str) -> str:
            return st.session_state.get(f"{name}_nonce")

        def _nonce_rotate(name: str) -> None:
            st.session_state[f"{name}_nonce"] = uuid.uuid4().hex

        if add_clicked:
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

            if phone_norm and len(phone_norm) != 10:
                st.error("Phone must be 10 digits or blank.")
            elif not business_name or not category:
                st.error("Provider and Category are required.")
            else:
                try:
                    _exec_with_retry(engine, "INSERT OR IGNORE INTO categories(name) VALUES(:n)", {"n": category})
                    if service:
                        _exec_with_retry(engine, "INSERT OR IGNORE INTO services(name) VALUES(:n)", {"n": service})

                    now = datetime.utcnow().isoformat(timespec="seconds")
                    computed = _computed_keywords_for(category, service, business_name) if service else ""
                    _exec_with_retry(
                        engine,
                        """
                        INSERT INTO vendors(category, service, business_name, contact_name, phone, address,
                                            website, notes, keywords, computed_keywords, created_at, updated_at, updated_by)
                        VALUES(:category, NULLIF(:service, ''), :business_name, :contact_name, :phone, :address,
                               :website, :notes, :keywords, :computed_keywords, :now, :now, :user)
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
                            "computed_keywords": computed,
                            "now": now,
                            "user": os.getenv("USER", "admin"),
                        },
                    )
                    st.session_state["add_last_done"] = add_nonce
                    st.success(f"Provider added: {business_name}")
                    _queue_add_form_reset()
                    _nonce_rotate("add")
                    st.rerun()
                except Exception as e:
                    st.error(f"Add failed: {e}")

    # RIGHT — Edit Provider
    with col_right:
        st.subheader("Edit Provider")

        if df_all.empty:
            st.info("No providers yet. Add a provider on the left.")
        else:
            edit_form_key = f"edit_provider_form_{st.session_state['edit_form_version']}"
            with st.form(edit_form_key, clear_on_submit=False):
                # Row 0: the selector (now INSIDE the form to align with spacer on the left)
                c0a, c0b = st.columns(2)
                with c0a:
                    st.selectbox(
                        "Select provider to edit (type to search)",
                        options=[None] + ids,
                        format_func=_fmt_provider,
                        key="edit_vendor_id",
                    )
                with c0b:
                    # keep row height symmetric; optional tiny caption space
                    st.caption("")

                # If selection changed, hydrate fields
                if st.session_state.get("edit_vendor_id") is not None:
                    if st.session_state.get("edit_last_loaded_id") != st.session_state.get("edit_vendor_id"):
                        row = id_to_row[int(st.session_state["edit_vendor_id"])]
                        st.session_state.update(
                            {
                                "edit_business_name": (row.get("business_name") or "").strip(),
                                "edit_category": (row.get("category") or "").strip(),
                                "edit_service": (row.get("service") or "").strip(),
                                "edit_contact_name": (row.get("contact_name") or "").strip(),
                                "edit_phone": (row.get("phone") or "").strip(),
                                "edit_address": (row.get("address") or "").strip(),
                                "edit_website": (row.get("website") or "").strip(),
                                "edit_notes": (row.get("notes") or "").strip(),
                                "edit_keywords": (row.get("keywords") or "").strip(),
                                "edit_row_updated_at": row.get("updated_at") or "",
                                "edit_last_loaded_id": st.session_state["edit_vendor_id"],
                            }
                        )

                # Row 1..: inputs laid out to mirror Add
                c1, c2 = st.columns(2)
                with c1:
                    st.text_input("Provider *", key="edit_business_name")

                    cats_choices = list_names(engine, "categories")
                    svcs_choices = list_names(engine, "services")
                    _edit_cat_options = [""] + (cats_choices or [])
                    _edit_svc_options = [""] + (svcs_choices or [])

                    if (st.session_state.get("edit_category") or "") not in _edit_cat_options:
                        st.session_state["edit_category"] = ""
                    if (st.session_state.get("edit_service") or "") not in _edit_svc_options:
                        st.session_state["edit_service"] = ""

                    st.selectbox("Category *", options=_edit_cat_options, key="edit_category", placeholder="Select category")
                    st.selectbox("Service (optional)", options=_edit_svc_options, key="edit_service")

                    st.text_input("Contact Name", key="edit_contact_name")
                    st.text_input("Phone (10 digits or blank)", key="edit_phone")
                with c2:
                    st.text_area("Address", height=80, key="edit_address")
                    st.text_input("Website (https://…)", key="edit_website")
                    st.text_area("Notes", height=100, key="edit_notes")
                    st.text_input("Keywords (comma separated)", key="edit_keywords")

                edited = st.form_submit_button("Save Provider Changes")

            def _nonce(name: str) -> str:
                return st.session_state.get(f"{name}_nonce")

            def _nonce_rotate(name: str) -> None:
                st.session_state[f"{name}_nonce"] = uuid.uuid4().hex

            if edited:
                edit_nonce = _nonce("edit")
                if st.session_state.get("edit_last_done") == edit_nonce:
                    st.info("Edit already processed.")
                    st.stop()

                vid = st.session_state.get("edit_vendor_id")
                if vid is None:
                    st.error("Select a provider first.")
                else:
                    bn = (st.session_state["edit_business_name"] or "").strip()
                    cat = (st.session_state["edit_category"] or "").strip()
                    svc = (st.session_state["edit_service"] or "").strip()
                    phone_norm = _normalize_phone(st.session_state["edit_phone"])
                    if phone_norm and len(phone_norm) != 10:
                        st.error("Phone must be 10 digits or blank.")
                    elif not bn:
                        st.error("Provider name is required.")
                    elif not cat:
                        st.error("Category is required. Pick one under 'Category Admin' if missing.")
                    else:
                        try:
                            _exec_with_retry(engine, "INSERT OR IGNORE INTO categories(name) VALUES(:n)", {"n": cat})
                            if svc:
                                _exec_with_retry(engine, "INSERT OR IGNORE INTO services(name) VALUES(:n)", {"n": svc})

                            prev_updated = st.session_state.get("edit_row_updated_at") or ""
                            now = datetime.utcnow().isoformat(timespec="seconds")
                            computed = _computed_keywords_for(cat, svc, bn) if svc else ""
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
                                       computed_keywords=:computed_keywords,
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
                                    "computed_keywords": computed,
                                    "now": now,
                                    "user": os.getenv("USER", "admin"),
                                    "id": int(vid),
                                    "prev_updated": prev_updated,
                                },
                            )
                            if (res.rowcount or 0) == 0:
                                st.warning("No changes applied (stale selection or already updated). Refresh and try again.")
                            else:
                                st.session_state["edit_last_done"] = edit_nonce
                                st.success(f"Provider updated: {bn}")
                                _queue_edit_form_reset()
                                _nonce_rotate("edit")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Update failed: {e}")

    # Bottom — Delete Provider
    st.markdown("---")
    st.subheader("Delete Provider")
    if df_all.empty:
        st.info("No providers to delete.")
    else:
        sel_label_del = st.selectbox(
            "Select provider to delete (type to search)",
            options=["— Select —"] + [_fmt_provider(i) for i in ids],
            key="delete_provider_label",
        )
        if sel_label_del != "— Select —":
            rev = {_fmt_provider(i): i for i in ids}
            st.session_state["delete_vendor_id"] = int(rev.get(sel_label_del))
        else:
            st.session_state["delete_vendor_id"] = None

        del_form_key = f"delete_provider_form_{st.session_state['delete_form_version']}"
        with st.form(del_form_key, clear_on_submit=False):
            deleted = st.form_submit_button("Delete Provider")

        if deleted:
            def _nonce(name: str) -> str:
                return st.session_state.get(f"{name}_nonce")

            def _nonce_rotate(name: str) -> None:
                st.session_state[f"{name}_nonce"] = uuid.uuid4().hex

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
                    res = _exec_with_retry(
                        engine,
                        "DELETE FROM vendors WHERE id=:id AND (updated_at=:prev_updated OR :prev_updated='')",
                        {"id": int(vid), "prev_updated": prev_updated},
                    )
                    if (res.rowcount or 0) == 0:
                        st.warning("No delete performed (stale selection). Refresh and try again.")
                    else:
                        st.session_state["delete_last_done"] = del_nonce
                        st.success("Provider deleted.")
                        _queue_delete_form_reset()
                        _nonce_rotate("delete")
                        st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")

# ---------- Category Admin
with _tabs[2]:
    st.caption("Category is required. Manage the reference list and reassign providers safely.")

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

    cats = list_names_normalized(engine, "categories")
    cat_opts = ["— Select —"] + cats

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
                st.write(f"In use by {cnt} provider(s).")
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
                    repl = st.selectbox("Reassign providers to…", options=repl_options, key="cat_reassign_to")
                    if st.button("Reassign providers then delete", key="cat_reassign_btn"):
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
    st.caption("Service is optional on providers. Manage the reference list here.")

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

    servs = list_names_normalized(engine, "services")
    svc_opts = ["— Select —"] + servs

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
            old = st.selectbox("Current", options=svc_opts, key="svc_old")
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
            tgt = st.selectbox("Service to delete", options=svc_opts, key="svc_del")
            if tgt == "— Select —":
                st.write("Select a service.")
            else:
                cnt = usage_count(engine, "service", tgt)
                st.write(f"In use by {cnt} provider(s).")
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
                    repl = st.selectbox("Reassign providers to…", options=repl_options, key="svc_reassign_to")
                    if st.button("Reassign providers then delete service", key="svc_reassign_btn"):
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

# ---------- System Tools / Help (Unified Maintenance) — ordered for operations flow
with _tabs[4]:
    st.header("System Tools / Help")

    # 1) Help & Documentation
    st.subheader("Help & Documentation")
    col1, col2 = st.columns([1, 0.15])
    with col1:
        if st.button("Resident Help / Tips"):
            st.markdown(_resolve_str("HELP_MD", "No help available."), unsafe_allow_html=True)
    with col2:
        st.caption("Read-first guidance for residents/read-only app.")

    col1, col2 = st.columns([1, 0.15])
    with col1:
        if st.button("Developer Help / Maintenance Manual"):
            st.markdown(_resolve_str("DEV_HELP_MD", "No developer manual found."), unsafe_allow_html=True)
    with col2:
        st.caption("Internal runbook: imports, probes, recovery, Git flow.")

    st.divider()

    # 2) Export backup
    st.subheader("Export All Providers (CSV Backup)")
    query = "SELECT * FROM vendors ORDER BY lower(business_name)"
    with engine.begin() as conn:
        full = pd.read_sql(sql_text(query), conn)
    export_df = full.copy()
    if "__spacer" in export_df.columns:
        export_df = export_df.drop(columns="__spacer")

    def _fmt_phone(x: str | int | None) -> str:
        s = re.sub(r"\D+", "", str(x or ""))
        return f"({s[0:3]}) {s[3:6]}-{s[6:10]}" if len(s) == 10 else (str(x or "").strip())

    if "phone" in export_df.columns:
        export_df["phone"] = export_df["phone"].apply(_fmt_phone)

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    st.download_button(
        "Export all providers (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"providers_{ts}.csv",
        mime="text/csv",
    )

    st.divider()

    # 3) CSV Restore (Append-only)
    st.subheader("CSV Restore (Append-only)")
    st.caption("Append-only; existing IDs are rejected. Use Dry run first to validate.")
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

    def _soft_trim(s: str | None) -> str:
        s = (s or "").strip()
        s = re.sub(r"[ \t]+", " ", s)
        return s

    def _fetch_existing_ids(engine: Engine, ids: Iterable[int]) -> set[int]:
        ids = [int(i) for i in ids if str(i).strip() != ""]
        if not ids:
            return set()
        params = {("i"+str(k)): ids[k] for k in range(len(ids))}
        q = f"SELECT id FROM vendors WHERE id IN ({','.join([':i'+str(k) for k in range(len(ids))])})"
        with engine.begin() as conn:
            rows = conn.execute(sql_text(q), params).fetchall()
        return set(int(r[0]) for r in rows)

    def _prepare_csv_for_append(
        engine: Engine,
        df_in: pd.DataFrame,
        *,
        normalize_phone: bool,
        trim_strings: bool,
        treat_missing_id_as_autoincrement: bool,
    ):
        with engine.begin() as conn:
            cols = conn.execute(sql_text("PRAGMA table_info(vendors)")).fetchall()
        vendors_cols = [c[1] for c in cols]
        insertable_cols = [c for c in df_in.columns if c in vendors_cols]
        if not insertable_cols:
            raise ValueError("CSV has no columns matching `vendors` table.")

        df = df_in[insertable_cols].copy()

        if trim_strings:
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].map(_soft_trim)

        if normalize_phone and "phone" in df.columns:
            df["phone"] = df["phone"].map(lambda x: _normalize_phone(str(x)))

        has_id = "id" in df.columns
        with_id_df = pd.DataFrame(columns=insertable_cols)
        without_id_df = pd.DataFrame(columns=insertable_cols)

        if has_id:
            mask_with_id = df["id"].astype(str).str.strip().ne("")
            with_id_df = df[mask_with_id].copy()
            without_id_df = df[~mask_with_id].copy()
        else:
            without_id_df = df.copy()

        rejected_ids: List[int] = []
        if has_id and not with_id_df.empty:
            existing = _fetch_existing_ids(engine, with_id_df["id"].astype(int).tolist())
            if existing:
                keep_mask = ~with_id_df["id"].astype(int).isin(existing)
                rejected_ids = sorted(list(existing))
                with_id_df = with_id_df[keep_mask].copy()

        if not treat_missing_id_as_autoincrement and not without_id_df.empty:
            without_id_df = pd.DataFrame(columns=insertable_cols)

        return with_id_df, without_id_df, rejected_ids, insertable_cols

    def _executemany_insert(engine: Engine, rows: List[Dict], cols: List[str]) -> int:
        if not rows:
            return 0
        now = datetime.utcnow().isoformat(timespec="seconds")
        for r in rows:
            r.setdefault("created_at", now)
            r.setdefault("updated_at", now)
            r.setdefault("updated_by", os.getenv("USER", "admin"))
            cat = (r.get("category") or "").strip()
            svc = (r.get("service") or "").strip()
            name = (r.get("business_name") or "").strip()
            if svc and not (r.get("computed_keywords") or "").strip():
                r["computed_keywords"] = _computed_keywords_for(cat, svc, name)

        col_list = ", ".join(cols)
        val_list = ", ".join([f":{c}" for c in cols])
        sql = f"INSERT INTO vendors ({col_list}) VALUES ({val_list})"
        with engine.begin() as conn:
            res = conn.execute(sql_text(sql), rows)
            return res.rowcount or 0

    def _execute_append_only(
        engine: Engine,
        with_id_df: pd.DataFrame,
        without_id_df: pd.DataFrame,
        insertable_cols: List[str],
    ) -> int:
        inserted = 0
        if not with_id_df.empty:
            rows = with_id_df.where(pd.notnull(with_id_df), None).to_dict(orient="records")
            inserted += _executemany_insert(engine, rows, insertable_cols)

        if not without_id_df.empty:
            cols_no_id = [c for c in insertable_cols if c != "id"]
            rows = without_id_df.where(pd.notnull(without_id_df), None).to_dict(orient="records")
            for r in rows:
                r.pop("id", None)
            inserted += _executemany_insert(engine, rows, cols_no_id)

        return inserted

    with st.expander("CSV Restore (Append-only) — Validate & Append"):
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
                        "rows_rejected_due_to_existing_id": rejected_ids or [],
                        "planned_inserts": int(planned_inserts),
                    }
                )

                if planned_inserts > 0:
                    preview_df = pd.concat([with_id_df, without_id_df], axis=0, ignore_index=True).head(20)
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)

                if dry_run:
                    st.success("Dry run complete. No changes applied.")
                else:
                    if planned_inserts == 0:
                        st.info("Nothing to insert (all rows rejected or CSV empty after filters).")
                    else:
                        try:
                            if st.button("Append CSV…", type="primary", key="append_csv_open"):
                                with st.dialog("Confirm CSV Append", width="small"):
                                    st.write("This will **append** rows from the uploaded file. This action is not reversible.")
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        if st.button("Cancel", key="append_cancel"):
                                            st.rerun()
                                    with c2:
                                        if st.button("Yes, append now", key="append_confirm"):
                                            inserted = _execute_append_only(engine, with_id_df, without_id_df, insertable_cols)
                                            st.success(f"Inserted {inserted} row(s). Rejected existing id(s): {rejected_ids or 'None'}")
                                            stats2 = _recompute_missing_computed_keywords(engine)
                                            if stats2.get("updated", 0):
                                                st.info(f"Computed keywords backfill after import: {stats2}")
                                            st.rerun()
                        except Exception:
                            if st.button("Append CSV (no modal)"):
                                inserted = _execute_append_only(engine, with_id_df, without_id_df, insertable_cols)
                                st.success(f"Inserted {inserted} row(s). Rejected existing id(s): {rejected_ids or 'None'}")
                                stats2 = _recompute_missing_computed_keywords(engine)
                                if stats2.get("updated", 0):
                                    st.info(f"Computed keywords backfill after import: {stats2}")
                                st.rerun()
            except Exception as e:
                st.error(f"CSV restore failed: {e}")
        else:
            st.info("Upload a CSV first, then open this panel to validate/append.")

    st.divider()

    # 4) Recompute Keywords (ALL rows)
    st.subheader("Recompute Keywords (ALL rows)")
    if st.button("Recompute now"):
        stats = recompute_keywords_all(engine)
        st.success(f"Recomputed for {stats['updated']} / {stats['checked']} rows.")

    st.divider()

    # 5) Show Unique Category / Service Pairs
    st.subheader("Show Unique Category / Service Pairs")
    if st.button("Run category/service analysis"):
        with engine.connect() as conn:
            rows = conn.execute(sql_text("""
                SELECT
                    TRIM(category) AS category,
                    TRIM(service)  AS service,
                    COUNT(*)       AS n
                FROM vendors
                GROUP BY category, service
                ORDER BY category COLLATE NOCASE, service COLLATE NOCASE;
            """)).fetchall()
        df_pairs = pd.DataFrame(rows, columns=["category", "service", "count"])
        st.dataframe(df_pairs, hide_index=True, use_container_width=True)

    st.divider()

    # 6) Quick Probes
    st.subheader("Quick Probes (read-only health checks)")
    _render_quick_probes_ui(engine)

    st.divider()

    # 7) Diagnostics / Debug
    st.subheader("Diagnostics / Debug")
    if st.button("Show Debug Info"):
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


if __name__ == "__main__":
    pass
