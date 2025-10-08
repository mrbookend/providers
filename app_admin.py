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
<<<<<<< HEAD
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

=======
# Page config & CSS (full width, no left margin; nowrap table)
# -----------------------------
PAGE_TITLE = st.secrets.get("page_title", "Vendors Admin") if hasattr(st, "secrets") else "Vendors Admin"
SIDEBAR_STATE = st.secrets.get("sidebar_state", "expanded") if hasattr(st, "secrets") else "expanded"

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

LEFT_PAD_PX = int(st.secrets.get("page_left_padding_px", 40)) if hasattr(st, "secrets") else 40

>>>>>>> parent of 9ad7caa (High-impact, low-risk wins admin.py)
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

<<<<<<< HEAD
# -----------------------------
# Admin auth (temporarily disabled)
# -----------------------------
st.session_state["admin_authed"] = True
# To re-enable later, restore the original form-based block and set
# DISABLE_ADMIN_PASSWORD="0" (and set ADMIN_PASSWORD) in Secrets.

# -----------------------------
# Engine builder (Direct Remote; hardened)
# -----------------------------
ALLOW_SQLITE_FALLBACK = _as_bool(_get_secret("ALLOW_SQLITE_FALLBACK", "0"), default=False)


def _normalize_remote_dsn(raw: str) -> str:
    """Ensure host DSN uses sqlite+libsql:// and secure=true."""
    dsn = (raw or "").strip()
=======
# -----------------------------
# Admin sign-in gate (robust)
# -----------------------------
ADMIN_PASSWORD = (st.secrets.get("ADMIN_PASSWORD") or os.getenv("ADMIN_PASSWORD") or "").strip()
# Dev bypass: set DISABLE_ADMIN_PASSWORD=1 in the environment to skip sign-in
if os.getenv("DISABLE_ADMIN_PASSWORD") == "1":
    st.session_state["auth_ok"] = True
# right after the bypass block in app_admin.py
if os.getenv("DISABLE_ADMIN_PASSWORD") == "1":
    st.info("⚠️ Admin password is DISABLED for this session (DEV).")

if os.getenv("DISABLE_ADMIN_PASSWORD") != "1" and (not isinstance(ADMIN_PASSWORD, str) or not ADMIN_PASSWORD):
    st.error("ADMIN_PASSWORD is not set in Secrets. Add it in Settings → Secrets.")
    st.stop()

if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False

if not st.session_state["auth_ok"]:
    st.subheader("Admin sign-in")
    pw = st.text_input("Password", type="password", key="admin_pw")
    if st.button("Sign in"):
        if (pw or "").strip() == ADMIN_PASSWORD:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# Small sanity check in Debug panel later:
# st.write({"FORCE_LOCAL": os.getenv("FORCE_LOCAL")})


# -----------------------------
# DB helpers
# -----------------------------

# --- CSV Restore helpers (ADD) ---

REQUIRED_VENDOR_COLUMNS = ["business_name", "category"]  # service optional
>>>>>>> parent of 9ad7caa (High-impact, low-risk wins admin.py)

    # libsql://host -> sqlite+libsql://host
    if dsn.startswith("libsql://"):
        dsn = "sqlite+libsql://" + dsn.split("://", 1)[1]

    # Guard against embedded DSNs; admin must use direct remote
    if dsn.startswith("sqlite+libsql:///"):
        raise ValueError(
            "Admin must use a direct remote DSN (sqlite+libsql://<host>?secure=true), not embedded."
        )

<<<<<<< HEAD
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
=======
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
# --- /CSV Restore helpers ---

def build_engine() -> tuple[Engine, dict]:
    """Use Embedded Replica for Turso (syncs to remote), else fallback to local."""
    info: dict = {}
>>>>>>> parent of 9ad7caa (High-impact, low-risk wins admin.py)

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
<<<<<<< HEAD
        dsn = _normalize_remote_dsn(url)
    except Exception as ex:
        _fail(str(ex))
=======
        # Normalize sync_url: embedded REQUIRES libsql:// (no sqlite+libsql, no ?secure=true)
        raw = url
        if raw.startswith("sqlite+libsql://"):
            host = raw.split("://", 1)[1].split("?", 1)[0]  # drop any ?secure=true
            sync_url = f"libsql://{host}"
        else:
            sync_url = raw  # already libsql://...
>>>>>>> parent of 9ad7caa (High-impact, low-risk wins admin.py)

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


<<<<<<< HEAD
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
=======
        # Prod: do NOT silently fall back
        st.error("Remote DB unavailable and FORCE_LOCAL is not set. Aborting to protect data.")
        raise




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
        "CREATE INDEX IF NOT EXISTS idx_vendors_kw  ON vendors(keywords)"
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
    # If it's not a clean 10 digits, show the original as-is
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

    for col in ["contact_name", "phone", "address", "website", "notes", "keywords", "service", "created_at", "updated_at", "updated_by"]:
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
>>>>>>> parent of 9ad7caa (High-impact, low-risk wins admin.py)

# -----------------------------
# UI: tables and editing
# -----------------------------
def _render_table(df: pd.DataFrame, title: str, max_height: int = 520) -> None:
    if df.empty:
        st.info(f"No rows in {title}.")
        return

<<<<<<< HEAD
    cols = list(df.columns)
    th = "".join(f"<th>{c}</th>" for c in cols)
    trs = []
    for _, r in df.iterrows():
        tds = "".join(f"<td>{str(r[c]) if pd.notna(r[c]) else ''}</td>" for c in cols)
        trs.append(f"<tr>{tds}</tr>")
    html = f"<h4>{title}</h4><table class='providers-grid'><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"
    components.html(html, height=max_height, scrolling=True)

=======

_tabs = st.tabs([
    "Browse Vendors",
    "Add / Edit / Delete Vendor",
    "Category Admin",
    "Service Admin",
    "Maintenance",
    "Debug",
])

# ---------- Browse
with _tabs[0]:
    df = load_df(engine)
    q = st.text_input(
        "",
        placeholder="Search — e.g., plumb returns any record with 'plumb' anywhere",
        label_visibility="collapsed",
    )
>>>>>>> parent of 9ad7caa (High-impact, low-risk wins admin.py)

def vendors_tab():
    st.subheader("Vendors")
    q = st.text_input("Search vendors (substring across common fields)", placeholder="e.g., plumb or 210-555-…")

<<<<<<< HEAD
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
=======
    # These two are already created in load_df(); remove if redundant:
    # df["notes_short"] = df["notes"].astype(str).str.replace("\n", " ").str.slice(0, 150)
    # df["keywords_short"] = df["keywords"].astype(str).str.replace("\n", " ").str.slice(0, 80)

    view_cols = [
        "id", "category", "service", "business_name", "contact_name", "phone_fmt",
        "address", "website", "notes", "keywords",
    ]
    vdf = _filter(df, q)[view_cols].rename(columns={"phone_fmt": "phone"})

    st.data_editor(
        vdf,
        use_container_width=False,
        hide_index=True,
        disabled=True,
        column_config={
            "business_name": st.column_config.TextColumn("Provider"),
            "website": st.column_config.TextColumn("website"),
            "notes": st.column_config.TextColumn(width=420),
            "keywords": st.column_config.TextColumn(width=300),
        },
    )

    st.download_button(
        "Download filtered view (CSV)",
        data=vdf.to_csv(index=False).encode("utf-8"),
        file_name="providers.csv",
        mime="text/csv",
    )

# ---------- Add/Edit/Delete Vendor

with _tabs[1]:
    st.subheader("Add Vendor")
    cats = list_names(engine, "categories")
    servs = list_names(engine, "services")

    with st.form("add_vendor"):
        col1, col2 = st.columns(2)
        with col1:
            business_name = st.text_input("Provider *")
            category = st.selectbox("Category *", options=cats, index=0 if cats else None, placeholder="Select category")
            service = st.selectbox("Service (optional)", options=[""] + servs, index=0)
            contact_name = st.text_input("Contact Name")
            phone = st.text_input("Phone (10 digits or blank)")
        with col2:
            address = st.text_area("Address", height=80)
            website = st.text_input("Website (https://…)")
            notes = st.text_area("Notes", height=100)
            keywords = st.text_input("Keywords (comma separated)")
        submitted = st.form_submit_button("Add Vendor")

    if submitted:
        if not business_name or not category:
            st.error("Business Name and Category are required.")
        else:
            phone_norm = _normalize_phone(phone)
            url = _sanitize_url(website)
            now = datetime.utcnow().isoformat(timespec="seconds")
            with engine.begin() as conn:
                conn.execute(sql_text(
                    """
                    INSERT INTO vendors(category, service, business_name, contact_name, phone, address, website, notes, keywords, created_at, updated_at, updated_by)
                    VALUES(:category, NULLIF(:service, ''), :business_name, :contact_name, :phone, :address, :website, :notes, :keywords, :now, :now, :user)
                    """
                ), {
                    "category": (category or "").strip(),
                    "service": (service or "").strip(),
                    "business_name": (business_name or "").strip(),
                    "contact_name": (contact_name or "").strip(),
                    "phone": phone_norm,
                    "address": (address or "").strip(),
                    "website": url,
                    "notes": (notes or "").strip(),
                    "keywords": (keywords or "").strip(),
                    "now": now,
                    "user": os.getenv("USER", "admin"),
                })
            st.success("Vendor added.")
            st.rerun()


    st.divider()
    st.subheader("Edit / Delete Vendor")

    df_all = load_df(engine)

    if df_all.empty:
        st.info("No vendors yet. Use 'Add Vendor' above to create your first record.")
>>>>>>> parent of 9ad7caa (High-impact, low-risk wins admin.py)
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

<<<<<<< HEAD
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
=======
# ---------- Category Admin
with _tabs[2]:
    st.caption("Category is required. Manage the reference list and reassign vendors safely.")
    cats = list_names(engine, "categories")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Add Category")
        new_cat = st.text_input("New category name")
        if st.button("Add Category"):
            if not new_cat.strip():
                st.error("Enter a name.")
            else:
                with engine.begin() as conn:
                    conn.execute(sql_text("INSERT OR IGNORE INTO categories(name) VALUES(:n)"), {"n": new_cat.strip()})
                st.success("Added (or already existed).")
                st.rerun()

        st.subheader("Rename Category")
        if cats:
            old = st.selectbox("Current", options=cats)
            new = st.text_input("New name", key="cat_rename")
            if st.button("Rename"):
                if not new.strip():
                    st.error("Enter a new name.")
                else:
                    with engine.begin() as conn:
                        conn.execute(sql_text("UPDATE categories SET name=:new WHERE name=:old"), {"new": new.strip(), "old": old})
                        conn.execute(sql_text("UPDATE vendors SET category=:new WHERE category=:old"), {"new": new.strip(), "old": old})
                    st.success("Renamed and reassigned.")
                    st.rerun()

    with colB:
        st.subheader("Delete / Reassign")
        if cats:
            tgt = st.selectbox("Category to delete", options=cats, key="cat_del")
            cnt = usage_count(engine, "category", tgt)
            st.write(f"In use by {cnt} vendor(s).")
            if cnt == 0:
                if st.button("Delete category (no usage)"):
                    with engine.begin() as conn:
                        conn.execute(sql_text("DELETE FROM categories WHERE name=:n"), {"n": tgt})
                    st.success("Deleted.")
                    st.rerun()
            else:
                repl_options = [c for c in cats if c != tgt]
                repl = st.selectbox("Reassign vendors to…", options=repl_options)
                if st.button("Reassign vendors then delete"):
                    with engine.begin() as conn:
                        conn.execute(sql_text("UPDATE vendors SET category=:r WHERE category=:t"), {"r": repl, "t": tgt})
                        conn.execute(sql_text("DELETE FROM categories WHERE name=:t"), {"t": tgt})
                    st.success("Reassigned and deleted.")
                    st.rerun()

# ---------- Service Admin
with _tabs[3]:
    st.caption("Service is optional on vendors. Manage the reference list here.")
    servs = list_names(engine, "services")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Add Service")
        new_s = st.text_input("New service name")
        if st.button("Add Service"):
            if not new_s.strip():
                st.error("Enter a name.")
            else:
                with engine.begin() as conn:
                    conn.execute(sql_text("INSERT OR IGNORE INTO services(name) VALUES(:n)"), {"n": new_s.strip()})
                st.success("Added (or already existed).")
                st.rerun()

        st.subheader("Rename Service")
        if servs:
            old = st.selectbox("Current", options=servs)
            new = st.text_input("New name", key="svc_rename")
            if st.button("Rename Service"):
                if not new.strip():
                    st.error("Enter a new name.")
                else:
                    with engine.begin() as conn:
                        conn.execute(sql_text("UPDATE services SET name=:new WHERE name=:old"), {"new": new.strip(), "old": old})
                        conn.execute(sql_text("UPDATE vendors SET service=:new WHERE service=:old"), {"new": new.strip(), "old": old})
                    st.success("Renamed and reassigned.")
                    st.rerun()

    with colB:
        st.subheader("Delete / Reassign")
        if servs:
            tgt = st.selectbox("Service to delete", options=servs, key="svc_del")
            cnt = usage_count(engine, "service", tgt)
            st.write(f"In use by {cnt} vendor(s).")
            if cnt == 0:
                if st.button("Delete service (no usage)"):
                    with engine.begin() as conn:
                        conn.execute(sql_text("DELETE FROM services WHERE name=:n"), {"n": tgt})
                    st.success("Deleted.")
                    st.rerun()
            else:
                repl_options = [s for s in servs if s != tgt]
                repl = st.selectbox("Reassign vendors to…", options=repl_options)
                if st.button("Reassign vendors then delete service"):
                    with engine.begin() as conn:
                        conn.execute(sql_text("UPDATE vendors SET service=:r WHERE service=:t"), {"r": repl, "t": tgt})
                        conn.execute(sql_text("DELETE FROM services WHERE name=:t"), {"t": tgt})
                    st.success("Reassigned and deleted.")
                    st.rerun()

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
            file_name="providers.csv",
            mime="text/csv",
        )
    with colB:
        st.download_button(
            "Export all vendors (digits-only phones)",
            data=full.to_csv(index=False).encode("utf-8"),
            file_name="providers_raw.csv",
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

 # Normalize phone numbers & Title Case (vendors + categories/services)
if st.button("Normalize phone numbers & Title Case (vendors + categories/services)"):
    def to_title(s: str | None) -> str:
        return ((s or "").strip()).title()

    TEXT_COLS_TO_TITLE = [
        "category", "service", "business_name", "contact_name",
        "address", "notes", "keywords",
    ]

    changed_vendors = 0
    with engine.begin() as conn:
        # --- vendors table ---
        rows = conn.execute(sql_text("SELECT * FROM vendors")).fetchall()
        for r in rows:
            row = dict(r._mapping) if hasattr(r, "_mapping") else dict(r)
            pid = int(row["id"])

            vals = {c: to_title(row.get(c)) for c in TEXT_COLS_TO_TITLE}
            vals["website"] = _sanitize_url((row.get("website") or "").strip())
            vals["phone"]   = _normalize_phone(row.get("phone") or "")
            vals["id"]      = pid

            conn.execute(sql_text("""
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
            """), vals)
            changed_vendors += 1

        # --- categories table: retitle + reconcile duplicates by case ---
        cat_rows = conn.execute(sql_text("SELECT name FROM categories")).fetchall()
        for (old_name,) in cat_rows:
            new_name = to_title(old_name)
            if new_name != old_name:
                conn.execute(sql_text("INSERT OR IGNORE INTO categories(name) VALUES(:n)"), {"n": new_name})
                conn.execute(sql_text("UPDATE vendors SET category=:new WHERE category=:old"),
                             {"new": new_name, "old": old_name})
                conn.execute(sql_text("DELETE FROM categories WHERE name=:old"), {"old": old_name})

        # --- services table: retitle + reconcile duplicates by case ---
        svc_rows = conn.execute(sql_text("SELECT name FROM services")).fetchall()
        for (old_name,) in svc_rows:
            new_name = to_title(old_name)
            if new_name != old_name:
                conn.execute(sql_text("INSERT OR IGNORE INTO services(name) VALUES(:n)"), {"n": new_name})
                conn.execute(sql_text("UPDATE vendors SET service=:new WHERE service=:old"),
                             {"new": new_name, "old": old_name})
                conn.execute(sql_text("DELETE FROM services WHERE name=:old"), {"old": old_name})

    st.success(f"Vendors normalized: {changed_vendors}. Categories/services retitled and reconciled.")



    # Backfill timestamps
    if st.button("Backfill created_at/updated_at when missing"):
        now = datetime.utcnow().isoformat(timespec="seconds")
        with engine.begin() as conn:
            conn.execute(
                sql_text(
                    "UPDATE vendors SET created_at=COALESCE(created_at, :now), updated_at=COALESCE(updated_at, :now)"
                ),
                {"now": now},
>>>>>>> parent of 9ad7caa (High-impact, low-risk wins admin.py)
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

<<<<<<< HEAD

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
=======
            def clean_soft(s: str | None) -> str:
                s = (s or "").strip()
                # collapse runs of spaces/tabs only; KEEP line breaks
                s = re.sub(r"[ \t]+", " ", s)
                return s

            for r in rows:
                pid = int(r[0])
                vals = {
                    "category":      clean_soft(r[1]),
                    "service":       clean_soft(r[2]),
                    "business_name": clean_soft(r[3]),
                    "contact_name":  clean_soft(r[4]),
                    "address":       clean_soft(r[5]),
                    "website":       _sanitize_url(clean_soft(r[6])),
                    "notes":         clean_soft(r[7]),  # preserves newlines
                    "keywords":      clean_soft(r[8]),
                    # leave phone unchanged here; or use _normalize_phone(r[9]) if you want to normalize now
                    "phone":         r[9],
                    "id":            pid,
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
            {"seq": r[0], "name": r[1], "unique": bool(r[2]), "origin": r[3], "partial": bool(r[4])}
            for r in idx_rows
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
    st.json({
        "vendors_columns": [c[1] for c in vendors_cols],
        "categories_columns": [c[1] for c in categories_cols],
        "services_columns": [c[1] for c in services_cols],
        "counts": counts,
        "vendors_indexes": vendors_indexes,
        "timestamp_nulls": {
            "created_at": int(created_at_nulls),
            "updated_at": int(updated_at_nulls),
        },
    })
>>>>>>> parent of 9ad7caa (High-impact, low-risk wins admin.py)
