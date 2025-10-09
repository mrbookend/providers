# admin.py
from __future__ import annotations

import os
import re
from typing import Any, Tuple, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine


# -------------------------------------------------------
# Optional dialect registration (sqlalchemy-libsql 0.2.x)
# -------------------------------------------------------
try:
    import sqlalchemy_libsql  # noqa: F401
except Exception:
    pass


# =======================================================
# Secrets helper
# =======================================================
def _get_secret(name: str, default: Any) -> Any:
    return st.secrets.get(name, default) if hasattr(st, "secrets") else os.getenv(name, default)


# =======================================================
# Page config & light CSS
# =======================================================
PAGE_TITLE = _get_secret("admin_page_title", _get_secret("page_title", "Providers — Admin"))
SIDEBAR_STATE = _get_secret("sidebar_state", "expanded")
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
      .stTextInput > label p {{
        margin-bottom: 0;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =======================================================
# Turso URL sanitizer + validator + gated fallback
# =======================================================
def _sanitize_turso_url(raw: str) -> str:
    """
    Accept various inputs and return clean libsql://HOST format only.
    Strips paths, querystrings and fragments that break libsql.
    """
    if not raw:
        return ""
    raw = raw.strip()

    if raw.lower().startswith("libsql://"):
        core = raw.split("://", 1)[1]
        core = core.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        return f"libsql://{core}"

    if raw.lower().startswith(("https://", "http://")):
        core = raw.split("://", 1)[1]
        core = core.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        return f"libsql://{core}"

    core = raw.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    return f"libsql://{core}" if core else ""


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
        if sync_url.lower().startswith("libsql://") and auth_token:
            try:
                eng = create_engine(
                    "sqlite+libsql:///vendors-embedded.db",
                    connect_args={"sync_url": sync_url, "auth_token": auth_token},
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
        else:
            info["remote_error"] = "Missing auth token or invalid URL after sanitization."

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
        "Check secrets or enable local fallback."
    )
    raise RuntimeError("No valid database connection available.")


engine, engine_info = build_engine()

# =======================================================
# Cache helpers & loaders
# =======================================================
@st.cache_data(ttl=5)
def _cache_key_for_engine(e: Engine) -> str:
    try:
        return e.url.render_as_string(hide_password=True)
    except Exception:
        return "engine"


@st.cache_data(ttl=600, show_spinner=False, hash_funcs={Engine: _cache_key_for_engine})
def load_vendors(e: Engine) -> pd.DataFrame:
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


@st.cache_data(ttl=600, show_spinner=False, hash_funcs={Engine: _cache_key_for_engine})
def load_taxonomy(e: Engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    with e.begin() as conn:
        cats = pd.read_sql(sql_text("SELECT id, name FROM categories ORDER BY name COLLATE NOCASE"), conn)
        svcs = pd.read_sql(sql_text("SELECT id, name FROM services ORDER BY name COLLATE NOCASE"), conn)
    return cats, svcs


# =======================================================
# Formatting helpers
# =======================================================
def _fmt_phone(v: Any) -> str:
    s = re.sub(r"\D+", "", str(v or ""))
    if len(s) == 10:
        return f"({s[0:3]}) {s[3:6]}-{s[6:10]}"
    return str(v or "")


def _sanitize_url(u: Optional[str]) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if not re.match(r"^https?://", u, re.I):
        u = "https://" + u
    return u


def _build_blob(df: pd.DataFrame) -> pd.DataFrame:
    if "_blob" in df.columns:
        return df
    parts = []
    for col in ["business_name", "category", "service", "contact_name", "phone", "address", "website", "notes", "keywords"]:
        if col in df.columns:
            parts.append(df[col].astype(str))
    if parts:
        df["_blob"] = pd.concat(parts, axis=1).agg(" ".join, axis=1).str.lower()
    else:
        df["_blob"] = ""
    return df


def _ensure_phone_fmt(df: pd.DataFrame) -> pd.DataFrame:
    if "phone_fmt" not in df.columns:
        if "phone" in df.columns:
            df = df.assign(phone_fmt=df["phone"].map(_fmt_phone))
        else:
            df = df.assign(phone_fmt="")
    return df


# =======================================================
# Data writes
# =======================================================
def upsert_vendor(
    e: Engine,
    *,
    id: Optional[int],
    category: str,
    service: str,
    business_name: str,
    contact_name: str,
    phone: str,
    address: str,
    website: str,
    notes: str,
    keywords: str,
    updated_by: str,
) -> int:
    """
    If id is None -> INSERT. Else UPDATE that row.
    Returns the id of the affected row.
    """
    website = _sanitize_url(website)
    with e.begin() as conn:
        if id is None:
            res = conn.execute(
                sql_text(
                    """
                    INSERT INTO vendors
                    (category, service, business_name, contact_name, phone, address, website, notes, keywords, created_at, updated_at, updated_by)
                    VALUES (:category, :service, :business_name, :contact_name, :phone, :address, :website, :notes, :keywords, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, :updated_by)
                    """
                ),
                {
                    "category": category.strip(),
                    "service": service.strip(),
                    "business_name": business_name.strip(),
                    "contact_name": (contact_name or "").strip(),
                    "phone": (phone or "").strip(),
                    "address": (address or "").strip(),
                    "website": website,
                    "notes": (notes or "").strip(),
                    "keywords": (keywords or "").strip(),
                    "updated_by": updated_by,
                },
            )
            new_id = res.lastrowid if hasattr(res, "lastrowid") else None
            if new_id is None:
                # Fallback: fetch last inserted rowid
                new_id = conn.execute(sql_text("SELECT last_insert_rowid()")).scalar()
            return int(new_id)
        else:
            conn.execute(
                sql_text(
                    """
                    UPDATE vendors
                       SET category     = :category,
                           service      = :service,
                           business_name= :business_name,
                           contact_name = :contact_name,
                           phone        = :phone,
                           address      = :address,
                           website      = :website,
                           notes        = :notes,
                           keywords     = :keywords,
                           updated_at   = CURRENT_TIMESTAMP,
                           updated_by   = :updated_by
                     WHERE id = :id
                    """
                ),
                {
                    "id": int(id),
                    "category": category.strip(),
                    "service": service.strip(),
                    "business_name": business_name.strip(),
                    "contact_name": (contact_name or "").strip(),
                    "phone": (phone or "").strip(),
                    "address": (address or "").strip(),
                    "website": website,
                    "notes": (notes or "").strip(),
                    "keywords": (keywords or "").strip(),
                    "updated_by": updated_by,
                },
            )
            return int(id)


def delete_vendor(e: Engine, *, id: int) -> None:
    with e.begin() as conn:
        conn.execute(sql_text("DELETE FROM vendors WHERE id = :id"), {"id": int(id)})


# =======================================================
# Load data & build local search cache
# =======================================================
vendors = load_vendors(engine)
vendors = _ensure_phone_fmt(vendors)
vendors = _build_blob(vendors)

cats_df, svcs_df = load_taxonomy(engine)
categories = sorted(cats_df["name"].dropna().unique().tolist()) if not cats_df.empty else []
services = sorted(svcs_df["name"].dropna().unique().tolist()) if not svcs_df.empty else []

# =======================================================
# Top: search + refresh
# =======================================================
top_l, top_r = st.columns([1, 1], gap="small")
with top_l:
    st.text_input(
        "Search",
        placeholder="Search providers… (press Enter)",
        label_visibility="collapsed",
        key="q",
    )
with top_r:
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

qq = (st.session_state.get("q") or "").strip().lower()
if qq:
    filtered = vendors[vendors["_blob"].str.contains(qq, regex=False, na=False)]
else:
    filtered = vendors

# Optional client-side filters for convenience (local lists)
sf_l, sf_r = st.columns([1, 1], gap="small")
with sf_l:
    cat_pick = st.selectbox("Category (optional)", ["(all)"] + categories, index=0)
with sf_r:
    svc_pick = st.selectbox("Service (optional)", ["(all)"] + services, index=0)

if cat_pick != "(all)":
    filtered = filtered[filtered["category"] == cat_pick]
if svc_pick != "(all)":
    filtered = filtered[filtered["service"] == svc_pick]

# =======================================================
# Table (plain text columns; no clickable styling)
# =======================================================
view_cols = [
    "business_name",
    "category",
    "service",
    "contact_name",
    "phone_fmt",
    "address",
    "website",
    "notes",
]
cols_present = [c for c in view_cols if c in filtered.columns]
grid_df = (
    filtered.reindex(columns=cols_present)
    .rename(columns={"business_name": "provider", "phone_fmt": "phone"})
)

st.dataframe(
    grid_df,
    use_container_width=True,
    column_config={
        "provider": st.column_config.TextColumn("Provider", width=240),
        "category": st.column_config.TextColumn("Category", width=160),
        "service": st.column_config.TextColumn("Service", width=160),
        "contact_name": st.column_config.TextColumn("Contact", width=180),
        "phone": st.column_config.TextColumn("Phone", width=140),
        "address": st.column_config.TextColumn("Address", width=260),
        "website": st.column_config.TextColumn("Website", width=220),
        "notes": st.column_config.TextColumn("Notes", width=420),
    },
    height=520,
)

st.download_button(
    "Download CSV (current view)",
    data=grid_df.to_csv(index=False).encode("utf-8"),
    file_name="providers_admin_view.csv",
    mime="text/csv",
)

st.divider()

# =======================================================
# Edit / Add panel
# =======================================================
st.subheader("Add or Edit a Provider")

# Row chooser (by provider)
provider_names = ["(new)"] + filtered["business_name"].astype(str).tolist()
pick = st.selectbox("Select provider to edit (or choose '(new)')", provider_names, index=0)

if pick != "(new)":
    row = filtered.loc[filtered["business_name"] == pick].iloc[0]
    current_id = int(row["id"]) if "id" in row else None
    init = {
        "category": row.get("category", ""),
        "service": row.get("service", ""),
        "business_name": row.get("business_name", ""),
        "contact_name": row.get("contact_name", ""),
        "phone": row.get("phone", ""),
        "address": row.get("address", ""),
        "website": row.get("website", ""),
        "notes": row.get("notes", ""),
        "keywords": row.get("keywords", ""),
    }
else:
    row = None
    current_id = None
    init = {
        "category": "",
        "service": "",
        "business_name": "",
        "contact_name": "",
        "phone": "",
        "address": "",
        "website": "",
        "notes": "",
        "keywords": "",
    }

# Simple form (fast; low risk)
with st.form("edit_add_vendor", clear_on_submit=False):
    f1, f2 = st.columns([1, 1], gap="small")
    with f1:
        category = st.text_input("Category", value=init["category"])
        service = st.text_input("Service", value=init["service"])
        business_name = st.text_input("Business name", value=init["business_name"])
        contact_name = st.text_input("Contact name", value=init["contact_name"])
        phone = st.text_input("Phone", value=init["phone"])
    with f2:
        address = st.text_area("Address", value=init["address"], height=80)
        website = st.text_input("Website", value=init["website"])
        notes = st.text_area("Notes", value=init["notes"], height=80)
        keywords = st.text_input("Keywords", value=init["keywords"])

    c1, c2, c3 = st.columns([0.2, 0.2, 0.6])
    do_save = c1.form_submit_button("Save")
    do_delete = c2.form_submit_button("Delete")

    if do_save:
        # Basic validations
        if not business_name.strip():
            st.error("Business name is required.")
        elif not category.strip():
            st.error("Category is required.")
        elif not service.strip():
            st.error("Service is required.")
        else:
            try:
                editor = str(_get_secret("ADMIN_UPDATED_BY", _get_secret("updated_by_default", "admin")) or "admin")
                new_id = upsert_vendor(
                    engine,
                    id=current_id,
                    category=category,
                    service=service,
                    business_name=business_name,
                    contact_name=contact_name,
                    phone=phone,
                    address=address,
                    website=website,
                    notes=notes,
                    keywords=keywords,
                    updated_by=editor,
                )
                st.success(f"Saved provider (id={new_id}).")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Save failed: {e}")

    if do_delete:
        if current_id is None:
            st.warning("Nothing to delete — select an existing provider first.")
        else:
            try:
                delete_vendor(engine, id=current_id)
                st.success(f"Deleted provider (id={current_id}).")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")

# =======================================================
# Optional maintenance/debug panel
# =======================================================
if "show_debug" not in st.session_state:
    st.session_state["show_debug"] = False

enable_debug = str(_get_secret("ADMIN_MAINTENANCE_ENABLE", _get_secret("READONLY_MAINTENANCE_ENABLE", "0"))).strip() == "1"

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
                counts = {
                    "vendors": conn.execute(sql_text("SELECT COUNT(*) FROM vendors")).scalar() or 0,
                    "categories": conn.execute(sql_text("SELECT COUNT(*) FROM categories")).scalar() or 0,
                    "services": conn.execute(sql_text("SELECT COUNT(*) FROM services")).scalar() or 0,
                }
            st.subheader("DB Probe")
            st.json(
                {
                    "vendors_columns": [c[1] for c in vendors_cols],
                    "counts": counts,
                }
            )
        except Exception as e:
            st.warning(f"Debug probe failed: {e}")
