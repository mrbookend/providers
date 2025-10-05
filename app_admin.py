from __future__ import annotations

import os
import re
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# -----------------------------
# Layout & Secrets
# -----------------------------
PAGE_TITLE = st.secrets.get("page_title", "Vendors Admin") if hasattr(st, "secrets") else "Vendors Admin"
PAGE_MAX_WIDTH_PX = int(st.secrets.get("page_max_width_px", 1400)) if hasattr(st, "secrets") else 1400
SIDEBAR_STATE = st.secrets.get("sidebar_state", "expanded") if hasattr(st, "secrets") else "expanded"

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

st.markdown(
    f"""
    <style>
      .block-container {{
        max-width: {PAGE_MAX_WIDTH_PX}px;
      }}
      /* keep data editor readable */
      div[data-testid="stDataFrame"] table {{
        white-space: normal; /* allow wrap */
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# DB Helpers
# -----------------------------

def _normalize_phone(val: str | None) -> str:
    if not val:
        return ""
    digits = re.sub(r"\D", "", str(val))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits if len(digits) == 10 else digits  # store as-is; UI can format


def _sanitize_url(url: str | None) -> str:
    if not url:
        return ""
    url = url.strip()
    if url and not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return url


def build_engine() -> Tuple[Engine, Dict]:
    """Prefer Turso (libSQL); fallback to local SQLite. Returns (engine, info)."""
    info = {}
    # Expect TURSO_DATABASE_URL like: sqlite+libsql://<host>?secure=true
    turso_url = os.getenv("TURSO_DATABASE_URL", st.secrets.get("TURSO_DATABASE_URL", ""))
    turso_token = os.getenv("TURSO_AUTH_TOKEN", st.secrets.get("TURSO_AUTH_TOKEN", ""))

    engine = None
    if turso_url and turso_token:
        try:
            engine = create_engine(turso_url, connect_args={"auth_token": turso_token}, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(sql_text("SELECT 1"))
            info.update({"using_remote": True, "sqlalchemy_url": turso_url, "dialect": engine.dialect.name, "driver": getattr(engine.dialect, "driver", "")})
            return engine, info
        except Exception as e:
            info["remote_error"] = str(e)

    # Fallback to local SQLite file
    engine = create_engine("sqlite:///vendors.db", pool_pre_ping=True)
    info.update({"using_remote": False, "sqlalchemy_url": "sqlite:///vendors.db", "dialect": engine.dialect.name, "driver": getattr(engine.dialect, "driver", "")})
    return engine, info


def ensure_schema(engine: Engine):
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
        # helpful indexes
        "CREATE INDEX IF NOT EXISTS idx_vendors_cat ON vendors(category)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_bus ON vendors(business_name)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_kw ON vendors(keywords)"
    ]
    with engine.begin() as conn:
        for s in stmts:
            conn.execute(sql_text(s))


def load_df(engine: Engine) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(sql_text("SELECT * FROM vendors ORDER BY lower(category), lower(business_name)"), conn)
    # ensure columns exist
    for col in ["contact_name", "phone", "address", "website", "notes", "keywords", "service", "created_at", "updated_at", "updated_by"]:
        if col not in df.columns:
            df[col] = ""
    # truncate long text for view
    df["notes_short"] = df.get("notes", "").astype(str).str.replace("\n", " ").str.slice(0, 150)
    df["keywords_short"] = df.get("keywords", "").astype(str).str.replace("\n", " ").str.slice(0, 80)
    return df


def list_names(engine: Engine, table: str) -> List[str]:
    with engine.begin() as conn:
        rows = conn.execute(sql_text(f"SELECT name FROM {table} ORDER BY lower(name)")).fetchall()
    return [r[0] for r in rows]


def usage_count(engine: Engine, col: str, name: str) -> int:
    with engine.begin() as conn:
        cnt = conn.execute(sql_text(f"SELECT COUNT(*) FROM vendors WHERE {col} = :n"), {"n": name}).scalar()
    return int(cnt or 0)

# -----------------------------
# UI
# -----------------------------
engine, engine_info = build_engine()
ensure_schema(engine)

st.title("Vendors — Admin")

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
    st.caption("Global search across key fields (case-insensitive; matches partial words).")
    q = st.text_input("Search", placeholder="e.g., plumb returns any record with 'plumb' anywhere")

    def _filter(_df: pd.DataFrame, q: str) -> pd.DataFrame:
        if not q:
            return _df
        qq = re.escape(q)
        mask = (
            _df["category"].str.contains(qq, case=False, na=False) |
            _df["service"].astype(str).str.contains(qq, case=False, na=False) |
            _df["business_name"].str.contains(qq, case=False, na=False) |
            _df["contact_name"].astype(str).str.contains(qq, case=False, na=False) |
            _df["phone"].astype(str).str.contains(qq, case=False, na=False) |
            _df["address"].astype(str).str.contains(qq, case=False, na=False) |
            _df["website"].astype(str).str.contains(qq, case=False, na=False) |
            _df["notes"].astype(str).str.contains(qq, case=False, na=False) |
            _df["keywords"].astype(str).str.contains(qq, case=False, na=False)
        )
        return _df[mask]

    view_cols = [
        "id", "category", "service", "business_name", "contact_name", "phone",
        "address", "website", "notes_short", "keywords_short"
    ]
    vdf = _filter(df, q)[view_cols].rename(columns={
        "notes_short": "notes",
        "keywords_short": "keywords"
    })

    st.data_editor(
        vdf,
        use_container_width=True,
        hide_index=True,
        disabled=True,
        column_config={
            "website": st.column_config.LinkColumn("website"),
            "notes": st.column_config.TextColumn(max_chars=150),
            "keywords": st.column_config.TextColumn(max_chars=80),
        },
    )

    st.download_button(
        "Download vendors.csv",
        data=vdf.to_csv(index=False).encode("utf-8"),
        file_name="vendors.csv",
        mime="text/csv",
    )

# ---------- Add/Edit/Delete
with _tabs[1]:
    st.subheader("Add Vendor")
    cats = list_names(engine, "categories")
    servs = list_names(engine, "services")

    with st.form("add_vendor"):
        col1, col2 = st.columns(2)
        with col1:
            business_name = st.text_input("Business Name *")
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
                    "category": category.strip(),
                    "service": (service or "").strip(),
                    "business_name": business_name.strip(),
                    "contact_name": contact_name.strip(),
                    "phone": phone_norm,
                    "address": address.strip(),
                    "website": url,
                    "notes": notes.strip(),
                    "keywords": keywords.strip(),
                    "now": now,
                    "user": os.getenv("USER", "admin"),
                })
            st.success("Vendor added.")
            st.experimental_rerun()

    st.divider()
    st.subheader("Edit / Delete Vendor")
    df_all = load_df(engine)
    id_list = df_all["id"].tolist()
    sel_id = st.selectbox("Select Vendor ID", options=id_list)
    row = df_all[df_all["id"] == sel_id].iloc[0]

    with st.form("edit_vendor"):
        col1, col2 = st.columns(2)
        with col1:
            business_name_e = st.text_input("Business Name *", row["business_name"])
            category_e = st.selectbox("Category *", options=cats, index=(cats.index(row["category"]) if row["category"] in cats else 0))
            service_e = st.selectbox("Service (optional)", options=[""] + servs, index=(([""] + servs).index(row["service"]) if str(row["service"]) in ([""] + servs) else 0))
            contact_name_e = st.text_input("Contact Name", row["contact_name"] or "")
            phone_e = st.text_input("Phone (10 digits or blank)", row["phone"] or "")
        with col2:
            address_e = st.text_area("Address", row["address"] or "", height=80)
            website_e = st.text_input("Website (https://…)", row["website"] or "")
            notes_e = st.text_area("Notes", row["notes"] or "", height=100)
            keywords_e = st.text_input("Keywords (comma separated)", row["keywords"] or "")
        c1, c2 = st.columns([1,1])
        update_btn = c1.form_submit_button("Save Changes")
        delete_btn = c2.form_submit_button("Delete Vendor", type="secondary")

    if update_btn:
        if not business_name_e or not category_e:
            st.error("Business Name and Category are required.")
        else:
            phone_norm = _normalize_phone(phone_e)
            url = _sanitize_url(website_e)
            now = datetime.utcnow().isoformat(timespec="seconds")
            with engine.begin() as conn:
                conn.execute(sql_text(
                    """
                    UPDATE vendors
                       SET category=:category, service=NULLIF(:service, ''), business_name=:business_name,
                           contact_name=:contact_name, phone=:phone, address=:address,
                           website=:website, notes=:notes, keywords=:keywords,
                           updated_at=:now, updated_by=:user
                     WHERE id=:id
                    """
                ), {
                    "category": category_e.strip(),
                    "service": (service_e or "").strip(),
                    "business_name": business_name_e.strip(),
                    "contact_name": contact_name_e.strip(),
                    "phone": phone_norm,
                    "address": address_e.strip(),
                    "website": url,
                    "notes": notes_e.strip(),
                    "keywords": keywords_e.strip(),
                    "now": now,
                    "user": os.getenv("USER", "admin"),
                    "id": int(sel_id),
                })
            st.success("Vendor updated.")
            st.experimental_rerun()

    if delete_btn:
        with engine.begin() as conn:
            conn.execute(sql_text("DELETE FROM vendors WHERE id=:id"), {"id": int(sel_id)})
        st.success("Vendor deleted.")
        st.experimental_rerun()

# ---------- Category Admin
with _tabs[2]:
    st.caption("Category is required. Use this tab to manage the reference list and reassign vendors safely.")
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
                st.experimental_rerun()

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
                    st.experimental_rerun()

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
                    st.experimental_rerun()
            else:
                repl_options = [c for c in cats if c != tgt]
                repl = st.selectbox("Reassign vendors to…", options=repl_options)
                if st.button("Reassign vendors then delete"):
                    with engine.begin() as conn:
                        conn.execute(sql_text("UPDATE vendors SET category=:r WHERE category=:t"), {"r": repl, "t": tgt})
                        conn.execute(sql_text("DELETE FROM categories WHERE name=:t"), {"t": tgt})
                    st.success("Reassigned and deleted.")
                    st.experimental_rerun()

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
                st.experimental_rerun()

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
                    st.experimental_rerun()

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
                    st.experimental_rerun()
            else:
                repl_options = [s for s in servs if s != tgt]
                repl = st.selectbox("Reassign vendors to…", options=repl_options)
                if st.button("Reassign vendors then delete service"):
                    with engine.begin() as conn:
                        conn.execute(sql_text("UPDATE vendors SET service=:r WHERE service=:t"), {"r": repl, "t": tgt})
                        conn.execute(sql_text("DELETE FROM services WHERE name=:t"), {"t": tgt})
                    st.success("Reassigned and deleted.")
                    st.experimental_rerun()

# ---------- Maintenance
with _tabs[4]:
    st.caption("One-click cleanups for legacy data.")

    if st.button("Normalize phones (digits only) & title-case business/contacts"):
        with engine.begin() as conn:
            rows = conn.execute(sql_text("SELECT id, phone, business_name, contact_name FROM vendors")).fetchall()
            for r in rows:
                pid = int(r[0])
                phone_norm = _normalize_phone(r[1] or "")
                bname = (r[2] or "").strip().title()
                cname = (r[3] or "").strip().title()
                conn.execute(sql_text(
                    "UPDATE vendors SET phone=:p, business_name=:b, contact_name=:c WHERE id=:id"
                ), {"p": phone_norm, "b": bname, "c": cname, "id": pid})
        st.success("Normalization complete.")

    if st.button("Backfill created_at/updated_at when missing"):
        now = datetime.utcnow().isoformat(timespec="seconds")
        with engine.begin() as conn:
            conn.execute(sql_text(
                "UPDATE vendors SET created_at=COALESCE(created_at, :now), updated_at=COALESCE(updated_at, :now)"
            ), {"now": now})
        st.success("Backfill complete.")

# ---------- Debug
with _tabs[5]:
    st.subheader("Status & Secrets (debug)")
    st.json(engine_info)

    with engine.begin() as conn:
        # columns
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

---

## app_readonly.py

# app_readonly.py — Read-only Vendors view
# Stable, no-AgGrid build. Uses native Streamlit components with a global search and clickable website links.

from __future__ import annotations

import os
import re
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# -----------------------------
# Layout & Secrets
# -----------------------------
PAGE_TITLE = st.secrets.get("page_title", "Vendors Directory") if hasattr(st, "secrets") else "Vendors Directory"
PAGE_MAX_WIDTH_PX = int(st.secrets.get("page_max_width_px", 1400)) if hasattr(st, "secrets") else 1400
SIDEBAR_STATE = st.secrets.get("sidebar_state", "collapsed") if hasattr(st, "secrets") else "collapsed"

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

st.markdown(
    f"""
    <style>
      .block-container {{ max-width: {PAGE_MAX_WIDTH_PX}px; }}
      div[data-testid="stDataFrame"] table {{ white-space: normal; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# DB
# -----------------------------

def build_engine() -> tuple[Engine, dict]:
    info = {}
    turso_url = os.getenv("TURSO_DATABASE_URL", st.secrets.get("TURSO_DATABASE_URL", ""))
    turso_token = os.getenv("TURSO_AUTH_TOKEN", st.secrets.get("TURSO_AUTH_TOKEN", ""))
    try:
        if turso_url and turso_token:
            engine = create_engine(turso_url, connect_args={"auth_token": turso_token}, pool_pre_ping=True)
            with engine.connect() as c:
                c.execute(sql_text("SELECT 1"))
            info.update({"using_remote": True, "sqlalchemy_url": turso_url, "dialect": "sqlite", "driver": "libsql"})
            return engine, info
    except Exception as e:
        info["remote_error"] = str(e)

    engine = create_engine("sqlite:///vendors.db", pool_pre_ping=True)
    info.update({"using_remote": False, "sqlalchemy_url": "sqlite:///vendors.db", "dialect": "sqlite", "driver": "pysqlite"})
    return engine, info


def load_df(engine: Engine) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(sql_text("SELECT * FROM vendors ORDER BY lower(category), lower(business_name)"), conn)
    for col in ["contact_name", "phone", "address", "website", "notes", "keywords", "service"]:
        if col not in df.columns:
            df[col] = ""
    df["notes"] = df["notes"].astype(str).str.replace("\n", " ").str.slice(0, 150)
    df["keywords"] = df["keywords"].astype(str).str.replace("\n", " ").str.slice(0, 80)
    return df

# -----------------------------
# UI
# -----------------------------
engine, engine_info = build_engine()
st.title("Vendors — Read‑only")

help_title = st.secrets.get("READONLY_HELP_TITLE", "Provider Help / Tips")
help_md = st.secrets.get("READONLY_HELP_MD", "")
with st.expander(help_title, expanded=False):
    st.markdown(help_md or "Use the search box to find vendors by name, category, service, or keywords.")

st.caption("Global search (case-insensitive; partial words ok). No per-column filters.")
q = st.text_input("Search", placeholder="e.g., plumb returns any record with 'plumb' anywhere")

raw = load_df(engine)

if q:
    qq = re.escape(q)
    m = (
        raw["category"].str.contains(qq, case=False, na=False) |
        raw["service"].astype(str).str.contains(qq, case=False, na=False) |
        raw["business_name"].str.contains(qq, case=False, na=False) |
        raw["contact_name"].astype(str).str.contains(qq, case=False, na=False) |
        raw["phone"].astype(str).str.contains(qq, case=False, na=False) |
        raw["address"].astype(str).str.contains(qq, case=False, na=False) |
        raw["website"].astype(str).str.contains(qq, case=False, na=False) |
        raw["notes"].astype(str).str.contains(qq, case=False, na=False) |
        raw["keywords"].astype(str).str.contains(qq, case=False, na=False)
    )
    df = raw[m]
else:
    df = raw

cols = ["category", "business_name", "contact_name", "phone", "address", "website", "notes", "service", "keywords", "id"]
view = df[cols]

st.data_editor(
    view,
    use_container_width=True,
    disabled=True,
    hide_index=True,
    column_config={
        "website": st.column_config.LinkColumn("website"),
        "phone": st.column_config.TextColumn("phone"),
        "notes": st.column_config.TextColumn(max_chars=150),
        "keywords": st.column_config.TextColumn(max_chars=80),
    },
)

st.download_button(
    "Download providers.csv",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name="providers.csv",
    mime="text/csv",
)

# Debug at bottom (optional)
if st.secrets.get("ADMIN_DEBUG", "false").lower() in ("1", "true", "yes"):  # opt-in
    st.divider()
    st.subheader("Status & Secrets (debug)")
    st.json(engine_info)

    with engine.begin() as conn:
        vendors_cols = conn.execute(sql_text("PRAGMA table_info(vendors)")).fetchall()
        counts = {"vendors": conn.execute(sql_text("SELECT COUNT(*) FROM vendors")).scalar() or 0}
    st.write("DB Probe")
    st.json({
        "vendors_columns": [c[1] for c in vendors_cols],
        "counts": counts,
    })

