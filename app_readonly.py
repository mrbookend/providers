from __future__ import annotations

import os
import re
from typing import Tuple, Dict

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# Register 'sqlite+libsql' dialect (must be after importing streamlit)
try:
    import sqlalchemy_libsql  # noqa: F401
except Exception:
    pass

# -----------------------------
# Page config & left padding
# -----------------------------
PAGE_TITLE = st.secrets.get("page_title", "Providers (Read-only)") if hasattr(st, "secrets") else "Providers (Read-only)"
SIDEBAR_STATE = st.secrets.get("sidebar_state", "expanded") if hasattr(st, "secrets") else "expanded"
st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

LEFT_PAD_PX = int(st.secrets.get("page_left_padding_px", 40)) if hasattr(st, "secrets") else 40
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
def build_engine() -> Tuple[Engine, Dict]:
    """Embedded replica to Turso. Fall back to local ONLY if FORCE_LOCAL=1."""
    info: Dict = {}
    url   = (st.secrets.get("TURSO_DATABASE_URL") or os.getenv("TURSO_DATABASE_URL") or "").strip()
    token = (st.secrets.get("TURSO_AUTH_TOKEN")   or os.getenv("TURSO_AUTH_TOKEN")   or "").strip()

    if not url:
        eng = create_engine("sqlite:///vendors.db", pool_pre_ping=True)
        info.update({
            "using_remote": False,
            "sqlalchemy_url": "sqlite:///vendors.db",
            "dialect": eng.dialect.name,
            "driver": getattr(eng.dialect, "driver", ""),
        })
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

        info.update({
            "using_remote": True,
            "strategy": "embedded_replica",
            "sqlalchemy_url": "sqlite+libsql:///vendors-embedded.db",
            "dialect": eng.dialect.name,
            "driver": getattr(eng.dialect, "driver", ""),
            "sync_url": sync_url,
        })
        return eng, info

    except Exception as e:
        info["remote_error"] = f"{e}"
        if os.getenv("FORCE_LOCAL") == "1":
            eng = create_engine("sqlite:///vendors.db", pool_pre_ping=True)
            info.update({
                "using_remote": False,
                "sqlalchemy_url": "sqlite:///vendors.db",
                "dialect": eng.dialect.name,
                "driver": getattr(eng.dialect, "driver", ""),
            })
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

def load_df(engine: Engine) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(sql_text("SELECT * FROM vendors ORDER BY lower(business_name)"), conn)
    # Robust: ensure columns exist if schema differs
    for col in ["category","service","business_name","contact_name","phone","address","website","notes","keywords","created_at","updated_at","updated_by"]:
        if col not in df.columns:
            df[col] = ""
    # Display-only phone format
    df["phone_fmt"] = df["phone"].apply(_format_phone)
    return df

# -----------------------------
# UI
# -----------------------------
# (No page title per your request)

_tabs = st.tabs(["Browse", "Debug"])

# -------- Browse
with _tabs[0]:
    df = load_df(engine)

    # Search (label hidden; placeholder carries "Search")
    st.caption("Global search across key fields (case-insensitive; partial words).")
    q = st.text_input(
        "",
        placeholder="Search — e.g., plumb returns any record with 'plumb' anywhere",
        label_visibility="collapsed",
    )

    def _filter(_df: pd.DataFrame, q: str) -> pd.DataFrame:
        if not q:
            return _df
        qq = re.escape(q)
        mask = (
            _df["category"].astype(str).str.contains(qq, case=False, na=False) |
            _df["service"].astype(str).str.contains(qq, case=False, na=False) |
            _df["business_name"].astype(str).str.contains(qq, case=False, na=False) |
            _df["contact_name"].astype(str).str.contains(qq, case=False, na=False) |
            _df["phone"].astype(str).str.contains(qq, case=False, na=False) |
            _df["address"].astype(str).str.contains(qq, case=False, na=False) |
            _df["website"].astype(str).str.contains(qq, case=False, na=False) |
            _df["notes"].astype(str).str.contains(qq, case=False, na=False) |
            _df["keywords"].astype(str).str.contains(qq, case=False, na=False)
        )
        return _df[mask]

    # Columns to show (HIDE 'keywords' by excluding it); use formatted phone
    view_cols = [
        "id", "category", "service", "business_name", "contact_name", "phone_fmt",
        "address", "website", "notes",
    ]
    grid_df = _filter(df, q)[view_cols].rename(columns={"phone_fmt": "phone"})

    # --- Make website clickable in a static rendered table ---
    def _linkify(u: str | None) -> str:
        u = (u or "").strip()
        if not u:
            return ""
        if not re.match(r"^https?://", u, re.I):
            u = "https://" + u
        disp = u.replace("https://", "").replace("http://", "")
        return f'<a href="{u}" target="_blank" rel="noopener noreferrer">{disp}</a>'

    # --- Build a Pandas Styler so we can FORCE WRAP and AUTO ROW HEIGHT ---
    styled = grid_df.style

    # Wrap long text for these columns; allow rows to grow automatically
    wrap_cols = [c for c in ["address", "website", "notes"] if c in grid_df.columns]
    if wrap_cols:
        styled = styled.set_properties(
            subset=wrap_cols,
            **{"white-space": "normal", "word-break": "break-word"}
        )

    # Startup column widths (tweak to taste)
    widths = {
        "category":       "140px",
        "service":        "160px",
        "business_name":  "240px",
        "contact_name":   "180px",
        "phone":          "140px",
        "address":        "260px",
        "website":        "220px",
        "notes":          "420px",
    }
    for col, w in widths.items():
        if col in grid_df.columns:
            styled = styled.set_properties(subset=[col], **{"min-width": w})

    # Clickable website links in the static table (don’t escape HTML here)
    if "website" in grid_df.columns:
        styled = styled.format({"website": _linkify}, escape=False)

    # Hide the index in the rendered table
    styled = styled.hide(axis="index")

    # Render as a static, styled table (guaranteed wrap + auto row height)
    st.dataframe(styled, use_container_width=True)

    # CSV download (formatted phone; still no 'keywords' column)
    st.download_button(
        "Download filtered view (CSV)",
        data=grid_df.to_csv(index=False).encode("utf-8"),
        file_name="providers_readonly.csv",
        mime="text/csv",
    )

# -------- Debug
with _tabs[1]:
    st.subheader("Status & Secrets (debug)")
    st.json(engine_info)

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
    st.json({
        "vendors_columns": [c[1] for c in vendors_cols],
        "categories_columns": [c[1] for c in categories_cols],
        "services_columns": [c[1] for c in services_cols],
        "counts": counts,
    })
