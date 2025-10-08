from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# Register 'sqlite+libsql' dialect (must be AFTER importing streamlit)
try:
    import sqlalchemy_libsql  # noqa: F401
except Exception:
    pass

# -----------------------------
# Page config & left padding (no title rendered)
# -----------------------------
PAGE_TITLE = (
    st.secrets.get("page_title", "Providers (Read-only)")
    if hasattr(st, "secrets")
    else "Providers (Read-only)"
)
SIDEBAR_STATE = (
    st.secrets.get("sidebar_state", "expanded")
    if hasattr(st, "secrets")
    else "expanded"
)
st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

LEFT_PAD_PX = int(st.secrets.get("page_left_padding_px", 20)) if hasattr(st, "secrets") else 20
st.markdown(
    f"""
    <style>
      [data-testid="stAppViewContainer"] .main .block-container {{
        padding-left: {LEFT_PAD_PX}px !important;
        padding-right: 0 !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Engine (embedded replica) + gated fallback
# -----------------------------
@st.cache_resource
def build_engine() -> tuple[Engine, dict[str, Any]]:
    """Embedded replica to Turso. Fall back to local ONLY if FORCE_LOCAL=1."""
    info: dict[str, Any] = {}
    url = (st.secrets.get("TURSO_DATABASE_URL") or os.getenv("TURSO_DATABASE_URL") or "").strip()
    token = (st.secrets.get("TURSO_AUTH_TOKEN") or os.getenv("TURSO_AUTH_TOKEN") or "").strip()

    if not url:
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

    try:
        # Normalize: embedded REQUIRES libsql:// (not sqlite+libsql://...?secure=true)
        raw = url
        if raw.startswith("sqlite+libsql://"):
            host = raw.split("://", 1)[1].split("?", 1)[0]
            sync_url = f"libsql://{host}"
        else:
            sync_url = raw

        eng = create_engine(
            "sqlite+libsql:///vendors-embedded.db",
            connect_args={"auth_token": token, "sync_url": sync_url},
            pool_pre_ping=True,
        )
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
        if os.getenv("FORCE_LOCAL") == "1":
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

        st.error("Remote DB unavailable and FORCE_LOCAL is not set. Aborting.")
        raise


engine, engine_info = build_engine()

# -----------------------------
# Helpers + data loader
# -----------------------------
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


@st.cache_data(ttl=5)
def _cache_key_for_engine(e: Engine) -> str:
    """Stable, secret-safe cache key for the SQLAlchemy Engine."""
    try:
        # Use the URL as a cache key but hide credentials
        return e.url.render_as_string(hide_password=True)
    except Exception:
        return "engine"

@st.cache_data(ttl=600, show_spinner=False, hash_funcs={Engine: _cache_key_for_engine})
def load_df(engine: Engine) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(
            sql_text("""
                SELECT id, category, service, business_name, contact_name,
                       phone, address, website, notes, keywords,
                       created_at, updated_at, updated_by
                FROM vendors
                ORDER BY business_name COLLATE NOCASE
            """),
            conn,
        )
    return df



# -----------------------------
# UI (single page; no title; no tabs)
# -----------------------------
if "show_debug" not in st.session_state:
    st.session_state["show_debug"] = False

# Top controls
top_l, top_r = st.columns([1, 1])
with top_l:
    q = st.text_input(
        "",
        placeholder="Search providers… (press Enter)",
        label_visibility="collapsed",
    )
with top_r:
    if st.button("Refresh data"):
        st.cache_data.clear()

df = load_df(engine)

# Filter quickly via precomputed blob
_filtered = df
qq = (q or "").strip().lower()
if qq:
    _filtered = _filtered[_filtered["_blob"].str.contains(qq, regex=False)]

# Columns to show (hide 'id' and 'keywords'); use formatted phone
view_cols = [
    "category",
    "service",
    "business_name",
    "contact_name",
    "phone_fmt",
    "address",
    "website",
    "notes",
]
# Ensure phone_fmt exists (format 10-digit US numbers), then build the grid data safely
if "phone_fmt" not in _filtered.columns:
    import re
    def _fmt_phone(v):
        s = re.sub(r"\D+", "", str(v or ""))
        if len(s) == 10:
            return f"({s[0:3]}) {s[3:6]}-{s[6:10]}"
        return str(v or "")
    _filtered = _filtered.assign(phone_fmt=_filtered["phone"].map(_fmt_phone))

# Columns to display (use phone_fmt, then rename to 'phone')
view_cols = [
    "business_name", "category", "service", "contact_name",
    "phone_fmt", "address", "website", "notes", "keywords"
]

# Guarded reindex avoids KeyError if any optional columns are missing
_cols = [c for c in view_cols if c in _filtered.columns]
grid_df = _filtered.reindex(columns=_cols).rename(
    columns={"business_name": "provider", "phone_fmt": "phone"}
)


# Render fast, virtualized table; clickable website links
st.dataframe(
    grid_df,
    use_container_width=True,
    column_config={
        "provider": st.column_config.TextColumn("Provider", width=240),
        "category": st.column_config.TextColumn("Category", width=140),
        "service": st.column_config.TextColumn("Service", width=160),
        "contact_name": st.column_config.TextColumn("Contact", width=180),
        "phone": st.column_config.TextColumn("Phone", width=140),
        "address": st.column_config.TextColumn("Address", width=260),
        "website": st.column_config.LinkColumn("Website", max_chars=40),
        "notes": st.column_config.TextColumn("Notes", width=420),
    },
    height=560,  # tweak if you want more/less visible rows
)

# CSV download (filtered view): formatted phones only
st.download_button(
    "Download CSV file of Providers",
    data=grid_df.to_csv(index=False).encode("utf-8"),
    file_name="providers.csv",
    mime="text/csv",
)

# -----------------------------
# Debug (button toggled at bottom)
# -----------------------------
st.divider()
btn_label = "Show debug" if not st.session_state["show_debug"] else "Hide debug"
if st.button(btn_label):
    st.session_state["show_debug"] = not st.session_state["show_debug"]
    st.rerun()

if st.session_state["show_debug"]:
    st.subheader("Status & Secrets (debug)")

    # mask sync_url in screenshots/logs
    safe_info = dict(engine_info)
    if isinstance(safe_info.get("sync_url"), str):
        s = safe_info["sync_url"]
        if len(s) > 20:
            safe_info["sync_url"] = s[:10] + "…•••…" + s[-8:]
    st.json(safe_info)

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
