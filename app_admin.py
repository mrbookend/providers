# -*- coding: utf-8 -*-
# app_admin.py - Providers Admin (direct remote; login-first; hardened engine)
from __future__ import annotations

import os
import hmac
import hashlib
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine
import sqlalchemy_libsql  # registers 'sqlite+libsql' dialect entrypoint


# -----------------------------
# Helpers for secrets / env
# -----------------------------
def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


def _as_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _norm(s: str, strip: bool) -> str:
    return s.strip() if strip else s


# -----------------------------
# Page config & basic styling
# -----------------------------
PAGE_TITLE = _get_secret("page_title", "Vendors Admin")
PAGE_MAX_WIDTH_PX = int(_get_secret("page_max_width_px", "2300") or "2300")
SIDEBAR_STATE = _get_secret("sidebar_state", "expanded")

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

st.markdown(
    f"""
    <style>
      .block-container {{ max-width: {PAGE_MAX_WIDTH_PX}px; }}
      table.providers-grid {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
      table.providers-grid th, table.providers-grid td {{
        border: 1px solid #ddd; padding: 6px; vertical-align: top; word-wrap: break-word; white-space: normal;
      }}
      table.providers-grid th {{ position: sticky; top: 0; background: #f7f7f7; z-index: 2; }}
      .wrap {{ white-space: normal; word-wrap: break-word; overflow-wrap: anywhere; }}
      .sticky-first {{ position: sticky; left: 0; background: #fff; z-index: 1; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Admin auth (single, robust; runs BEFORE engine creation)
# -----------------------------
if not st.session_state.get("admin_authed", False):
    with st.form("admin_login", clear_on_submit=False):
        pw = st.text_input("Admin password", type="password", key="admin_pw")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            # Optional dev bypass
            if _as_bool(_get_secret("DISABLE_ADMIN_PASSWORD", "0")):
                st.session_state["admin_authed"] = True
                st.success("Signed in (bypass).")
                st.rerun()

            raw_secret = _get_secret("ADMIN_PASSWORD", "")
            if not raw_secret:
                st.error("ADMIN_PASSWORD is not set in Secrets or Env.")
                st.stop()

            strip_cfg = _as_bool(_get_secret("ADMIN_PASSWORD_STRIP", "1"), True)
            typed = _norm(pw or "", strip_cfg)
            stored = str(raw_secret)

            ok = False
            if stored.startswith("sha256:"):
                stored_hash = stored.split("sha256:", 1)[1]
                typed_hash = hashlib.sha256(typed.encode("utf-8")).hexdigest()
                ok = hmac.compare_digest(typed_hash, stored_hash)
            else:
                ok = hmac.compare_digest(typed, _norm(stored, strip_cfg))

            if _as_bool(_get_secret("DEBUG_ADMIN_AUTH", "0")):
                st.caption(
                    f"Auth debug: strip={strip_cfg}, "
                    f"len(typed)={len(typed)}, len(config)={len(stored)}, "
                    f"eq_plain={typed == stored}, eq_strip={typed == _norm(stored, True)}"
                )

            if not ok:
                st.error("Invalid password.")
                st.stop()

            st.session_state["admin_authed"] = True
            st.success("Signed in.")
            st.rerun()
else:
    st.caption("Signed in as admin.")


# -----------------------------
# Engine builder (Direct Remote; hardened)
# -----------------------------
ALLOW_SQLITE_FALLBACK = _as_bool(_get_secret("ALLOW_SQLITE_FALLBACK", "0"), default=False)


def _normalize_remote_dsn(raw: str) -> str:
    """Ensure host DSN uses sqlite+libsql:// and secure=true."""
    dsn = (raw or "").strip()

    # libsql://host -> sqlite+libsql://host
    if dsn.startswith("libsql://"):
        dsn = "sqlite+libsql://" + dsn.split("://", 1)[1]

    # Guard against embedded DSNs; admin must use direct remote
    if dsn.startswith("sqlite+libsql:///"):
        raise ValueError(
            "Admin must use a direct remote DSN (sqlite+libsql://<host>?secure=true), not embedded."
        )

    # Ensure secure=true
    if "secure=" not in dsn.lower():
        dsn += ("&secure=true" if "?" in dsn else "?secure=true")
    return dsn


def build_engine() -> Tuple[Engine, Dict[str, Any]]:
    """
    Direct-remote builder for admin. Enforces TLS on remote DSN, probes SELECT 1,
    and optionally falls back to local SQLite if ALLOW_SQLITE_FALLBACK=true.
    """
    url = _get_secret("TURSO_DATABASE_URL", "") or ""
    token = _get_secret("TURSO_AUTH_TOKEN", "") or ""

    info: Dict[str, Any] = {
        "using_remote": False,
        "dialect": "",
        "driver": "",
        "sqlalchemy_url": "",
        "strategy": "",
    }

    def _fail(msg: str):
        st.error(
            "SQLAlchemy couldn't load the 'libsql' dialect or connect. "
            "Verify sqlalchemy-libsql==0.2.0 is installed and DSN uses sqlite+libsql://. "
            f"Details: {msg}"
        )
        st.stop()

    def _fallback_sqlite(reason: str) -> Tuple[Engine, Dict[str, Any]]:
        if not ALLOW_SQLITE_FALLBACK:
            _fail(reason + " Also, SQLite fallback is disabled. Or set ALLOW_SQLITE_FALLBACK=true (dev only).")
        local_path = os.getenv("LOCAL_SQLITE_PATH", "vendors.db")
        e = create_engine(f"sqlite:///{local_path}")
        info.update({"sqlalchemy_url": f"sqlite:///{local_path}", "strategy": "local_sqlite"})
        try:
            info["dialect"] = e.dialect.name
            info["driver"] = getattr(e.dialect, "driver", "")
        except Exception:
            pass
        return e, info

    if not url or not token:
        return _fallback_sqlite("TURSO_DATABASE_URL / TURSO_AUTH_TOKEN missing.")

    try:
        dsn = _normalize_remote_dsn(url)
    except Exception as ex:
        _fail(str(ex))

    try:
        e = create_engine(
            dsn,
            connect_args={"auth_token": token},
            pool_pre_ping=True,
            pool_recycle=180,
        )
        # Probe the connection now
        with e.connect() as c:
            c.exec_driver_sql("SELECT 1")

        info.update({"using_remote": True, "sqlalchemy_url": dsn, "strategy": "direct"})
        try:
            info["dialect"] = e.dialect.name
            info["driver"] = getattr(e.dialect, "driver", "")
        except Exception:
            pass
        return e, info
    except Exception as ex:
        name = type(ex).__name__
        return _fallback_sqlite(f"Turso init failed ({name}: {ex})")


@st.cache_resource
def get_engine_and_info() -> Tuple[Engine, Dict[str, Any]]:
    # Cache the engine object; created only after successful auth
    return build_engine()


engine, engine_info = get_engine_and_info()

# -----------------------------
# Data helpers (exec path tuned for libsql)
# -----------------------------
@st.cache_data(ttl=30)
def fetch_df(sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    SELECT helper. For named params, use text() to avoid libsql positional quirks.
    """
    try:
        with engine.connect() as conn:
            if params:
                result = conn.execute(sql_text(sql), params)
            else:
                result = conn.exec_driver_sql(sql)
            rows = result.fetchall()
            cols = result.keys()
        return pd.DataFrame(rows, columns=list(cols))
    except Exception as ex:
        st.error(f"DB query failed: {type(ex).__name__}: {ex}")
        raise


def exec_sql(sql: str, params: Optional[Dict[str, Any]] = None) -> int:
    """
    DML/DDL helper. Returns rowcount when available.
    """
    try:
        with engine.begin() as conn:
            if params:
                result = conn.execute(sql_text(sql), params)
            else:
                result = conn.exec_driver_sql(sql)
        try:
            fetch_df.clear()
        except Exception:
            pass
        return getattr(result, "rowcount", -1)
    except Exception as ex:
        st.error(f"DB execute failed: {type(ex).__name__}: {ex}")
        raise


# -----------------------------
# UI: tables and editing
# -----------------------------
def _render_table(df: pd.DataFrame, title: str, max_height: int = 520) -> None:
    if df.empty:
        st.info(f"No rows in {title}.")
        return

    cols = list(df.columns)
    th = "".join(f"<th>{c}</th>" for c in cols)
    trs = []
    for _, r in df.iterrows():
        tds = "".join(f"<td>{str(r[c]) if pd.notna(r[c]) else ''}</td>" for c in cols)
        trs.append(f"<tr>{tds}</tr>")
    html = f"<h4>{title}</h4><table class='providers-grid'><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"
    components.html(html, height=max_height, scrolling=True)


def vendors_tab():
    st.subheader("Vendors")
    q = st.text_input("Search vendors (substring across common fields)", placeholder="e.g., plumb or 210-555-…")

    if q:
        like = f"%{q.lower()}%"
        sql = """
            SELECT id, business_name, category, service, contact_name, phone, address, website, notes, keywords,
                   created_at, updated_at, updated_by
            FROM vendors
            WHERE lower(business_name) LIKE :like
               OR lower(category)      LIKE :like
               OR lower(service)       LIKE :like
               OR lower(contact_name)  LIKE :like
               OR lower(phone)         LIKE :like
               OR lower(address)       LIKE :like
               OR lower(website)       LIKE :like
               OR lower(notes)         LIKE :like
               OR lower(keywords)      LIKE :like
            ORDER BY business_name COLLATE NOCASE
            LIMIT 500
        """
        df = fetch_df(sql, {"like": like})
    else:
        df = fetch_df(
            """
            SELECT id, business_name, category, service, contact_name, phone, address, website, notes, keywords,
                   created_at, updated_at, updated_by
            FROM vendors
            ORDER BY business_name COLLATE NOCASE
            LIMIT 500
            """
        )

    _render_table(df, "Vendors (first 500)")

    # Editor
    st.markdown("---")
    st.subheader("Edit / Create Vendor")

    # Build selector list
    df_small = fetch_df("SELECT id, business_name FROM vendors ORDER BY business_name COLLATE NOCASE LIMIT 500")
    options = [("Create new…", None)] + [(f"{int(r.id)} — {r.business_name}", int(r.id)) for _, r in df_small.iterrows()]
    choice = st.selectbox(
        "Choose a row to edit, or create new:",
        options,
        format_func=lambda x: x[0] if isinstance(x, tuple) else x,
    )
    choice_id = choice[1] if isinstance(choice, tuple) else None

    # Load existing row if editing
    row = None
    if choice_id is not None:
        row_df = fetch_df("SELECT * FROM vendors WHERE id = :id", {"id": choice_id})
        if not row_df.empty:
            row = row_df.iloc[0].to_dict()

    def _val(name: str, default: str = "") -> str:
        return str((row or {}).get(name, default) or "")

    with st.form("vendor_edit", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            business_name = st.text_input("Business name", _val("business_name"))
            category = st.text_input("Category", _val("category"))
            service = st.text_input("Service", _val("service"))
            contact_name = st.text_input("Contact name", _val("contact_name"))
        with c2:
            phone = st.text_input("Phone", _val("phone"))
            address = st.text_input("Address", _val("address"))
            website = st.text_input("Website", _val("website"))
        with c3:
            keywords = st.text_area("Keywords", _val("keywords"), height=94)
            notes = st.text_area("Notes", _val("notes"), height=94)

        submitted_update = st.form_submit_button("Save (Create or Update)")

    if submitted_update:
        if choice_id is None:
            # Create
            rc = exec_sql(
                """
                INSERT INTO vendors (business_name, category, service, contact_name, phone, address, website, notes, keywords, created_at, updated_at, updated_by)
                VALUES (:business_name, :category, :service, :contact_name, :phone, :address, :website, :notes, :keywords, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'admin')
                """,
                {
                    "business_name": business_name.strip(),
                    "category": category.strip(),
                    "service": service.strip(),
                    "contact_name": contact_name.strip(),
                    "phone": phone.strip(),
                    "address": address.strip(),
                    "website": website.strip(),
                    "notes": notes.strip(),
                    "keywords": keywords.strip(),
                },
            )
            st.success(f"Inserted {rc} row(s).")
        else:
            # Update
            rc = exec_sql(
                """
                UPDATE vendors
                   SET business_name = :business_name,
                       category = :category,
                       service   = :service,
                       contact_name = :contact_name,
                       phone = :phone,
                       address = :address,
                       website = :website,
                       notes = :notes,
                       keywords = :keywords,
                       updated_at = CURRENT_TIMESTAMP,
                       updated_by = 'admin'
                 WHERE id = :id
                """,
                {
                    "id": choice_id,
                    "business_name": business_name.strip(),
                    "category": category.strip(),
                    "service": service.strip(),
                    "contact_name": contact_name.strip(),
                    "phone": phone.strip(),
                    "address": address.strip(),
                    "website": website.strip(),
                    "notes": notes.strip(),
                    "keywords": keywords.strip(),
                },
            )
            st.success(f"Updated {rc} row(s).")

    # Delete section
    with st.form("vendor_delete", clear_on_submit=True):
        del_id = st.number_input("Delete vendor by ID", min_value=0, step=1, value=0)
        submitted_del = st.form_submit_button("Delete")
    if submitted_del and del_id > 0:
        rc = exec_sql("DELETE FROM vendors WHERE id = :id", {"id": int(del_id)})
        st.success(f"Deleted {rc} row(s).")


def categories_tab():
    st.subheader("Categories")
    df = fetch_df("SELECT id, name FROM categories ORDER BY name COLLATE NOCASE")
    _render_table(df, "Categories", max_height=360)

    with st.form("cat_edit", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("New category name")
            submitted_add = st.form_submit_button("Add")
        with c2:
            cat_id = st.number_input("Delete category by ID", min_value=0, step=1, value=0)
            submitted_del = st.form_submit_button("Delete")
    if submitted_add and name.strip():
        rc = exec_sql("INSERT INTO categories (name) VALUES (:name)", {"name": name.strip()})
        st.success(f"Inserted {rc} row(s).")
    if submitted_del and cat_id > 0:
        rc = exec_sql("DELETE FROM categories WHERE id = :id", {"id": int(cat_id)})
        st.success(f"Deleted {rc} row(s).")


def services_tab():
    st.subheader("Services")
    df = fetch_df("SELECT id, name FROM services ORDER BY name COLLATE NOCASE")
    _render_table(df, "Services", max_height=360)

    with st.form("srv_edit", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("New service name")
            submitted_add = st.form_submit_button("Add")
        with c2:
            srv_id = st.number_input("Delete service by ID", min_value=0, step=1, value=0)
            submitted_del = st.form_submit_button("Delete")
    if submitted_add and name.strip():
        rc = exec_sql("INSERT INTO services (name) VALUES (:name)", {"name": name.strip()})
        st.success(f"Inserted {rc} row(s).")
    if submitted_del and srv_id > 0:
        rc = exec_sql("DELETE FROM services WHERE id = :id", {"id": int(srv_id)})
        st.success(f"Deleted {rc} row(s).")


def sql_tab():
    st.subheader("SQL Console (admin)")
    st.caption("For quick diagnostics. SELECT shows a table; DML/DDL returns rowcount. Use with care.")
    sql = st.text_area("SQL", height=160, placeholder="e.g., SELECT * FROM vendors LIMIT 10")
    run = st.button("Run")
    if run and sql.strip():
        try:
            if sql.strip().lower().startswith("select"):
                df = fetch_df(sql)
                st.dataframe(df, use_container_width=True)
            else:
                rc = exec_sql(sql)
                st.success(f"Rowcount (if available): {rc}")
        except Exception:
            st.stop()


def debug_tab():
    st.subheader("Debug / Status")
    st.json(engine_info)
    try:
        with engine.connect() as conn:
            vendors_cols = [r[1] for r in conn.exec_driver_sql("PRAGMA table_info(vendors)").fetchall()]
            categories_cols = [r[1] for r in conn.exec_driver_sql("PRAGMA table_info(categories)").fetchall()]
            services_cols = [r[1] for r in conn.exec_driver_sql("PRAGMA table_info(services)").fetchall()]
            cnts_row = conn.exec_driver_sql(
                "SELECT (SELECT COUNT(1) FROM vendors) AS vendors, "
                "(SELECT COUNT(1) FROM categories) AS categories, "
                "(SELECT COUNT(1) FROM services) AS services"
            ).fetchone()
        st.json(
            {
                "vendors_columns": vendors_cols,
                "categories_columns": categories_cols,
                "services_columns": services_cols,
                "counts": {
                    "vendors": int(cnts_row[0]),
                    "categories": int(cnts_row[1]),
                    "services": int(cnts_row[2]),
                },
            }
        )
    except Exception as e:
        st.error(f"Probe failed: {e}")


# -----------------------------
# Main
# -----------------------------
def main():
    tabs = st.tabs(["Vendors", "Categories", "Services", "SQL", "Debug"])
    with tabs[0]:
        vendors_tab()
    with tabs[1]:
        categories_tab()
    with tabs[2]:
        services_tab()
    with tabs[3]:
        sql_tab()
    with tabs[4]:
        debug_tab()


if __name__ == "__main__":
    main()
