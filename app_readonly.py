from __future__ import annotations

import os
import re
from typing import Any, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# Ensure libsql dialect is registered (must be imported once).
try:
    import sqlalchemy_libsql  # noqa: F401
except Exception:
    pass


# =============================================================================
# Page config & styling
# =============================================================================
def _get_secret(name: str, default: Any = None) -> Any:
    return (getattr(st, "secrets", {}) or {}).get(name, default)

PAGE_TITLE = _get_secret("page_title", "Providers — Read-only")
SIDEBAR_STATE = _get_secret("sidebar_state", "expanded")
PAGE_MAX_W = int(_get_secret("page_max_width_px", 1600))
LEFT_PAD_PX = int(_get_secret("page_left_padding_px", 16))

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

st.markdown(
    f"""
    <style>
      .block-container {{
        padding-left: {LEFT_PAD_PX}px !important;
        padding-right: 0 !important;
        max-width: {PAGE_MAX_W}px !important;
      }}
      /* Make the table feel a touch denser without harming readability */
      [data-testid="stDataFrame"] tbody tr td {{
        padding-top: 6px !important;
        padding-bottom: 6px !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Turso URL sanitizer / validator
# =============================================================================
def _strip_invisibles(s: str) -> str:
    # Remove zero-width and whitespace/control characters that often sneak in.
    return "".join(ch for ch in s if ch.isprintable()).strip()

def _normalize_sync_url(raw: str) -> str:
    """
    Accepts a variety of inputs and returns a clean libsql:// URL or "".
    Prevents libsql Rust client panics due to invalid URI chars.
    """
    if not raw:
        return ""
    s = _strip_invisibles(raw).strip(" '\"\n\r\t`")
    # If someone pasted a SQLAlchemy-style URL, convert to libsql sync URL.
    if s.startswith("sqlite+libsql://"):
        # take host[:port][/path] before '?'
        host = s.split("://", 1)[1].split("?", 1)[0]
        s = f"libsql://{host}"
    # If they pasted a https://turso.link URL (rare), reject for embedded.
    if s.startswith("https://"):
        # embedded replica requires libsql://
        s = s.replace("https://", "libsql://", 1)

    # Basic validation: scheme + host; turso hosts have dots and dashes.
    if not s.startswith("libsql://"):
        return ""
    host_part = s[len("libsql://") :]
    if not host_part or any(c in host_part for c in ' <>|"\\^{}[]'):
        return ""
    return s


# =============================================================================
# Engine (embedded replica) with guarded fallback
# =============================================================================
@st.cache_resource
def build_engine() -> Tuple[Engine, dict[str, Any]]:
    """
    Prefer embedded replica backed by Turso. If invalid/absent and FORCE_LOCAL=="1",
    fall back to a local SQLite file.
    """
    info: dict[str, Any] = {}
    url_raw = _get_secret("TURSO_DATABASE_URL", os.getenv("TURSO_DATABASE_URL", ""))
    token = _get_secret("TURSO_AUTH_TOKEN", os.getenv("TURSO_AUTH_TOKEN", "")) or ""
    sync_url = _normalize_sync_url(url_raw)

    force_local = str(os.getenv("FORCE_LOCAL", _get_secret("FORCE_LOCAL", "0"))).strip() == "1"

    def _local_engine() -> Tuple[Engine, dict[str, Any]]:
        eng = create_engine(
            "sqlite:///vendors.db",
            pool_pre_ping=True,
            connect_args={"check_same_thread": False},
        )
        meta = {
            "using_remote": False,
            "sqlalchemy_url": "sqlite:///vendors.db",
            "dialect": eng.dialect.name,
            "driver": getattr(eng.dialect, "driver", ""),
        }
        return eng, meta

    # No remote URL: maybe local?
    if not sync_url:
        if force_local:
            return _local_engine()
        st.error("Remote database URL is invalid or missing, and FORCE_LOCAL is not enabled.")
        raise RuntimeError("Missing or invalid TURSO_DATABASE_URL")

    # Try embedded replica
    try:
        eng = create_engine(
            "sqlite+libsql:///vendors-embedded.db",
            connect_args={"auth_token": token, "sync_url": sync_url},
            pool_pre_ping=True,
        )
        # Probe connection early so failures are surfaced up-front
        with eng.connect() as c:
            c.exec_driver_sql("select 1;")

        info.update(
            {
                "using_remote": True,
                "strategy": "embedded_replica",
                "sqlalchemy_url": "sqlite+libsql:///vendors-embedded.db",
                "dialect": eng.dialect.name,
                "driver": getattr(eng.dialect, "driver", ""),
                "sync_url": sync_url,  # masked in debug panel
            }
        )
        return eng, info

    except Exception as e:
        info["remote_error"] = f"{e}"
        if force_local:
            return _local_engine()
        st.error("Remote DB connection failed and FORCE_LOCAL is not set. Aborting.")
        raise


engine, engine_info = build_engine()


# =============================================================================
# Helpers
# =============================================================================
def _format_phone(val: str | None) -> str:
    s = re.sub(r"\D", "", str(val or ""))
    if len(s) == 10:
        return f"({s[0:3]}) {s[3:6]}-{s[6:10]}"
    return (val or "").strip()

def _sanitize_url(u: str | None) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if not re.match(r"^https?://", u, re.I):
        u = "https://" + u
    return u


# Cache key for Engine (stable & secret-safe)
@st.cache_data(ttl=5)
def _cache_key_for_engine(e: Engine) -> str:
    try:
        return e.url.render_as_string(hide_password=True)
    except Exception:
        return "engine"


# =============================================================================
# Data loading
#   * Only pull columns we render
#   * Enrich with phone_fmt and _blob (cached), so search never races
# =============================================================================
@st.cache_data(ttl=600, show_spinner=False, hash_funcs={Engine: _cache_key_for_engine})
def load_df(e: Engine) -> pd.DataFrame:
    with e.begin() as conn:
        df = pd.read_sql(
            sql_text(
                """
                SELECT id, category, service, business_name, contact_name,
                       phone, address, website, notes, keywords,
                       created_at, updated_at, updated_by
                FROM vendors
                ORDER BY business_name COLLATE NOCASE
                """
            ),
            conn,
        )

    # Enrich: formatted phone & search blob (lowercased concatenation)
    if "phone" in df.columns and "phone_fmt" not in df.columns:
        df["phone_fmt"] = df["phone"].map(_format_phone)

    # Keep website as plain text (no clickable column per request), but sanitize
    if "website" in df.columns:
        df["website"] = df["website"].map(_sanitize_url)

    # Build a precomputed search blob to keep filtering vectorized & fast
    cols_for_blob = [
        c for c in [
            "business_name", "category", "service", "contact_name",
            "phone_fmt", "address", "website", "notes", "keywords"
        ] if c in df.columns
    ]
    def _to_text(x: Any) -> str:
        return "" if pd.isna(x) else str(x)
    df["_blob"] = (
        df[cols_for_blob]
        .astype(str)
        .applymap(_to_text)
        .agg(" ".join, axis=1)
        .str.lower()
    )

    return df


df = load_df(engine)
# --- Build a lowercase search blob once (guarded) ---
if "_blob" not in df.columns:
    _parts = []
    for col in ["business_name", "category", "service", "contact_name", "phone", "address", "website", "notes", "keywords"]:
        if col in df.columns:
            _parts.append(df[col].astype(str))
    if _parts:
        df["_blob"] = (
            pd.concat(_parts, axis=1)
            .agg(" ".join, axis=1)
            .str.lower()
        )
    else:
        df["_blob"] = ""


# =============================================================================
# UI – controls
# =============================================================================
# Initialize query from URL once
try:
    if "q" not in st.session_state:
        st.session_state["q"] = st.query_params.get("q", "")
except Exception:
    if "q" not in st.session_state:
        st.session_state["q"] = ""

top_l, top_r = st.columns([1, 1])
with top_l:
    st.text_input(
        "Search",  # non-empty label to fix accessibility warnings
        placeholder="Search providers… (press Enter)",
        label_visibility="collapsed",
        key="q",
    )
with top_r:
    if st.button("Refresh data"):
        st.cache_data.clear()

# Keep URL in sync with current query (nice deep-linking)
try:
    current_qp = dict(st.query_params)
    q_now = st.session_state.get("q", "")
    if q_now:
        if current_qp.get("q") != q_now:
            st.query_params["q"] = q_now
    else:
        if "q" in current_qp:
            del st.query_params["q"]
except Exception:
    pass


# =============================================================================
# Filtering & view model
# =============================================================================
_filtered = df
qq = (st.session_state.get("q") or "").strip().lower()
if qq:
    _filtered = _filtered[_filtered["_blob"].str.contains(qq, regex=False, na=False)]

# Render columns (plain text; no clickable phone/website)
view_cols = [
    "business_name", "category", "service", "contact_name",
    "phone_fmt", "address", "website", "notes"
]
_cols = [c for c in view_cols if c in _filtered.columns]

grid_df = (
    _filtered
    .reindex(columns=_cols)
    .rename(columns={"business_name": "provider", "phone_fmt": "phone"})
)

column_widths = _get_secret(
    "COLUMN_WIDTHS_PX_READONLY",
    {
        "provider": 220, "category": 160, "service": 160, "contact_name": 180,
        "phone": 140, "address": 260, "website": 220, "notes": 420
    },
)

label_overrides = _get_secret("READONLY_COLUMN_LABELS", {"provider": "Provider"})

column_config = {
    "provider": st.column_config.TextColumn(label_overrides.get("provider", "Provider"),
                                            width=column_widths.get("provider", 220)),
    "category": st.column_config.TextColumn("Category", width=column_widths.get("category", 160)),
    "service": st.column_config.TextColumn("Service", width=column_widths.get("service", 160)),
    "contact_name": st.column_config.TextColumn("Contact", width=column_widths.get("contact_name", 180)),
    "phone": st.column_config.TextColumn("Phone", width=column_widths.get("phone", 140)),
    "address": st.column_config.TextColumn("Address", width=column_widths.get("address", 260)),
    "website": st.column_config.TextColumn("Website", width=column_widths.get("website", 220)),
    "notes": st.column_config.TextColumn("Notes", width=column_widths.get("notes", 420)),
}

st.dataframe(
    grid_df,
    use_container_width=True,
    hide_index=True,
    column_config=column_config,
    height=560,
)

# Download filtered view
st.download_button(
    "Download CSV file of Providers",
    data=grid_df.to_csv(index=False).encode("utf-8"),
    file_name="providers.csv",
    mime="text/csv",
)


# =============================================================================
# Debug / maintenance (optional, controlled by secret)
# =============================================================================
if _get_secret("READONLY_MAINTENANCE_ENABLE", "0") == "1":
    st.divider()
    if "show_debug" not in st.session_state:
        st.session_state["show_debug"] = False

    btn_label = "Show debug" if not st.session_state["show_debug"] else "Hide debug"
    if st.button(btn_label):
        st.session_state["show_debug"] = not st.session_state["show_debug"]
        st.rerun()

    if st.session_state["show_debug"]:
        st.subheader("Status & Secrets (debug)")

        # Mask sync_url tail
        safe_info = dict(engine_info)
        if isinstance(safe_info.get("sync_url"), str):
            s = safe_info["sync_url"]
            if len(s) > 20:
                safe_info["sync_url"] = s[:10] + "…•••…" + s[-8:]
        st.json(safe_info)

        # PRAGMA probes
        try:
            with engine.begin() as conn:
                vendors_cols = conn.execute(sql_text("PRAGMA table_info(vendors)")).fetchall()
                categories_cols = conn.execute(sql_text("PRAGMA table_info(categories)")).fetchall()
                services_cols = conn.execute(sql_text("PRAGMA table_info(services)")).fetchall()
                counts = {
                    "vendors": conn.execute(sql_text("SELECT COUNT(*) FROM vendors")).scalar() or 0,
                    "categories": conn.execute(sql_text("SELECT COUNT(*) FROM categories")).scalar() or 0,
                    "services": conn.execute(sql_text("SELECT COUNT(*) FROM services")).scalar() or 0,
                }
                vendors_indexes = conn.execute(sql_text("PRAGMA index_list(vendors)")).fetchall()
        except Exception as ex:
            vendors_cols = categories_cols = services_cols = vendors_indexes = []
            counts = {"vendors": 0, "categories": 0, "services": 0}
            st.warning(f"Debug probes failed: {ex}")

        st.subheader("DB Probe")
        st.json(
            {
                "vendors_columns": [c[1] for c in vendors_cols],
                "categories_columns": [c[1] for c in categories_cols],
                "services_columns": [c[1] for c in services_cols],
                "counts": counts,
                "vendors_indexes": [
                    {
                        "seq": ix[0],
                        "name": ix[1],
                        "unique": bool(ix[2]),
                        "origin": ix[3],
                        "partial": bool(ix[4]) if len(ix) > 4 else False,
                    }
                    for ix in vendors_indexes
                ],
            }
        )
