# app_readonly.py
from __future__ import annotations

import os
import re
from typing import Any, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# Try to register the sqlite+libsql dialect (works with sqlalchemy-libsql==0.2.x)
try:
    import sqlalchemy_libsql  # noqa: F401
except Exception:
    pass


# =========================================================
# Page config and simple styling
# =========================================================
def _get_secret(name: str, default: Any) -> Any:
    return st.secrets.get(name, default) if hasattr(st, "secrets") else os.getenv(name, default)


PAGE_TITLE = _get_secret("page_title", "Providers — Read-only")
SIDEBAR_STATE = _get_secret("sidebar_state", "collapsed")  # "expanded" | "collapsed"
PAGE_MAX_WIDTH = int(_get_secret("page_max_width_px", 2300))
LEFT_PAD_PX = int(_get_secret("page_left_padding_px", 20))

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

st.markdown(
    f"""
    <style>
      .main .block-container {{
        padding-left: {LEFT_PAD_PX}px !important;
        padding-right: 0 !important;
        max-width: {PAGE_MAX_WIDTH}px !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Turso engine builder (sanitizer + validator + gated fallback)
# =========================================================
def _sanitize_turso_url(raw: str) -> str:
    """
    Accepts various incoming forms and returns a clean libsql://host path
    that libsql can handle. Strips querystrings/paths that cause InvalidUri.
    """
    if not raw:
        return ""
    raw = raw.strip()

    # If it already starts with libsql://, strip everything except host
    if raw.lower().startswith("libsql://"):
        core = raw.split("://", 1)[1]
        # remove path / query / fragment
        core = core.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        return f"libsql://{core}"

    # Some people paste https URLs (browser endpoints) — keep only host
    if raw.lower().startswith("https://") or raw.lower().startswith("http://"):
        core = raw.split("://", 1)[1]
        core = core.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        return f"libsql://{core}"

    # Bare host
    core = raw.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    if core:
        return f"libsql://{core}"

    return ""


@st.cache_resource
def build_engine() -> Tuple[Engine, dict[str, Any]]:
    """
    Preferred: embedded replica engine to Turso (sqlite+libsql:///… with sync_url & token).
    Fallback to local sqlite:///vendors.db ONLY if FORCE_LOCAL="1".
    """
    info: dict[str, Any] = {}
    raw_url = str(_get_secret("TURSO_DATABASE_URL", "") or "").strip()
    auth_token = str(_get_secret("TURSO_AUTH_TOKEN", "") or "").strip()
    force_local = str(_get_secret("FORCE_LOCAL", os.getenv("FORCE_LOCAL", "0"))).strip()

    if raw_url:
        sync_url = _sanitize_turso_url(raw_url)
        if not sync_url.lower().startswith("libsql://"):
            info["remote_error"] = "Sanitized URL is not libsql://…"
        elif not auth_token:
            info["remote_error"] = "TURSO_AUTH_TOKEN missing"
        else:
            try:
                eng = create_engine(
                    "sqlite+libsql:///vendors-embedded.db",
                    connect_args={"sync_url": sync_url, "auth_token": auth_token},
                    pool_pre_ping=True,
                )
                # Validate connectivity early
                with eng.connect() as c:
                    c.exec_driver_sql("select 1;")

                info.update(
                    {
                        "using_remote": True,
                        "strategy": "embedded_replica",
                        "sqlalchemy_url": "sqlite+libsql:///vendors-embedded.db",
                        "dialect": eng.dialect.name,
                        "driver": getattr(eng.dialect, "driver", ""),
                        "sync_url": sync_url,
                    }
                )
                return eng, info
            except Exception as e:
                info["remote_error"] = f"{e}"

    # Reaching here means remote path failed or not configured
    if force_local == "1":
        eng = create_engine("sqlite:///vendors.db", pool_pre_ping=True)
        info.update(
            {
                "using_remote": False,
                "sqlalchemy_url": "sqlite:///vendors.db",
                "dialect": eng.dialect.name,
                "driver": getattr(eng.dialect, "driver", ""),
            }
        )
        return eng, info

    st.error(
        "Remote DB unavailable (or misconfigured), and FORCE_LOCAL is not set to '1'. "
        "Please set secrets properly or enable local fallback."
    )
    raise RuntimeError("No valid database connection available.")


engine, engine_info = build_engine()

# =========================================================
# Cache helpers + data loader
# =========================================================
@st.cache_data(ttl=5)
def _cache_key_for_engine(e: Engine) -> str:
    try:
        return e.url.render_as_string(hide_password=True)
    except Exception:
        return "engine"


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
    return df


df = load_df(engine)

# --- Build phone_fmt once (guarded)
if "phone_fmt" not in df.columns:
    def _fmt_phone(v: Any) -> str:
        s = re.sub(r"\D+", "", str(v or ""))
        if len(s) == 10:
            return f"({s[0:3]}) {s[3:6]}-{s[6:10]}"
        return str(v or "")
    df["phone_fmt"] = df["phone"].map(_fmt_phone) if "phone" in df.columns else ""

# --- Build a lowercase search blob once (guarded)
if "_blob" not in df.columns:
    parts = []
    for col in [
        "business_name",
        "category",
        "service",
        "contact_name",
        "phone",
        "address",
        "website",
        "notes",
        "keywords",
    ]:
        if col in df.columns:
            parts.append(df[col].astype(str))
    if parts:
        df["_blob"] = pd.concat(parts, axis=1).agg(" ".join, axis=1).str.lower()
    else:
        df["_blob"] = ""

# =========================================================
# UI — top controls and fast local search
# =========================================================
top_l, top_r = st.columns([1, 1])
with top_l:
    q = st.text_input(
        "Search",
        placeholder="Search providers… (press Enter)",
        label_visibility="collapsed",
        key="q",
    )
with top_r:
    if st.button("Refresh data"):
        st.cache_data.clear()

qq = (st.session_state.get("q") or "").strip().lower()
if qq:
    filtered = df[df["_blob"].str.contains(qq, regex=False, na=False)]
else:
    filtered = df

# =========================================================
# Grid render (no clickable phone/website)
# =========================================================
# Columns to display (and order). Keep them defensive.
view_cols = [
    "business_name",
    "category",
    "service",
    "contact_name",
    "phone_fmt",
    "address",
    "website",
    "notes",
    # "keywords"  # omit from view; still in CSV export via filtered if desired
]
cols_present = [c for c in view_cols if c in filtered.columns]
grid_df = filtered.reindex(columns=cols_present).rename(
    columns={
        "business_name": "provider",
        "phone_fmt": "phone",
    }
)

st.dataframe(
    grid_df,
    use_container_width=True,
    column_config={
        "provider": st.column_config.TextColumn(_get_secret("READONLY_COLUMN_LABELS", {}).get("provider", "Provider"), width=220),
        "category": st.column_config.TextColumn("Category", width=160),
        "service": st.column_config.TextColumn("Service", width=160),
        "contact_name": st.column_config.TextColumn("Contact", width=180),
        "phone": st.column_config.TextColumn("Phone", width=140),
        "address": st.column_config.TextColumn("Address", width=260),
        "website": st.column_config.TextColumn("Website", width=220),
        "notes": st.column_config.TextColumn("Notes", width=420),
    },
    height=560,
)

# CSV download (filtered, with formatted phone)
st.download_button(
    "Download CSV file of Providers",
    data=grid_df.to_csv(index=False).encode("utf-8"),
    file_name="providers.csv",
    mime="text/csv",
)

# =========================================================
# Optional maintenance/debug panel
# =========================================================
if "show_debug" not in st.session_state:
    st.session_state["show_debug"] = False

enable_debug = str(_get_secret("READONLY_MAINTENANCE_ENABLE", "0")).strip() == "1"

if enable_debug:
    st.divider()
    btn_label = "Show debug" if not st.session_state["show_debug"] else "Hide debug"
    if st.button(btn_label):
        st.session_state["show_debug"] = not st.session_state["show_debug"]
        st.rerun()

    if st.session_state["show_debug"]:
        st.subheader("Status & Secrets (debug)")

        safe_info = dict(engine_info)
        if isinstance(safe_info.get("sync_url"), str):
            s = safe_info["sync_url"]
            if len(s) > 20:
                safe_info["sync_url"] = s[:10] + "…•••…" + s[-8:]
        st.json(safe_info)

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
            st.subheader("DB Probe")
            st.json(
                {
                    "vendors_columns": [c[1] for c in vendors_cols],
                    "categories_columns": [c[1] for c in categories_cols],
                    "services_columns": [c[1] for c in services_cols],
                    "counts": counts,
                }
            )
        except Exception as e:
            st.warning(f"Debug probe failed: {e}")
