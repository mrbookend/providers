# app_admin.py — Providers Admin (tuned to match read-only app conventions)
# Key traits:
# - Wide layout via secrets (page_title, page_max_width_px, sidebar_state)
# - Turso/libSQL first; optional guarded SQLite fallback for local dev
# - Simple password gate (secrets["ADMIN_PASSWORD"]) before UI renders
# - Browse: HTML table with real pixel widths from secrets, cell wrapping, client-side search (Python) + server-side sort
# - Vendors: Add / Edit / Delete with validation; phone normalization; service optional
# - Categories/Services Admin: add/rename/delete, usage counts, orphan surfacing, reassign-then-delete workflow
# - Maintenance: light fix-ups (title-case, phone normalize, audit fields), CSV exports (filtered + full)
# - Debug: compact button at bottom to reveal engine + schema snapshot

from __future__ import annotations

import os
import re
import html
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# -----------------------------
# Page layout FIRST
# -----------------------------

def _read_secret_early(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)

PAGE_TITLE = _read_secret_early("page_title", "Providers Admin")
PAGE_MAX_WIDTH_PX = int(_read_secret_early("page_max_width_px", 2000))
SIDEBAR_STATE = _read_secret_early("sidebar_state", "expanded")

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
      .dim {{ opacity: 0.9; }}
      .muted {{ color: #666; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Secrets helpers
# -----------------------------

def _get_secret(name: str, default: Optional[str | int | bool | dict] = None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)

# Column widths (px) for the admin browse table
# Prefer COLUMN_WIDTHS_PX_ADMIN; fallback to COLUMN_WIDTHS_PX
RAW_COL_WIDTHS = _get_secret("COLUMN_WIDTHS_PX_ADMIN") or _get_secret("COLUMN_WIDTHS_PX") or {
    "id": 40,
    "business_name": 160,
    "category": 170,
    "service": 110,
    "contact_name": 110,
    "phone": 110,
    "address": 200,
    "website": 180,
    "notes": 220,
    "keywords": 120,
}

ADMIN_HELP_TITLE = _get_secret("ADMIN_HELP_TITLE", "Admin Tips")
ADMIN_HELP_MD = _get_secret("ADMIN_HELP_MD", "")
ADMIN_DEBUG = str(_get_secret("ADMIN_DEBUG", "0")).lower() in ("1", "true", "yes")
ALLOW_SQLITE_FALLBACK = str(_get_secret("ALLOW_SQLITE_FALLBACK", "0")).lower() in ("1", "true", "yes")

# -----------------------------
# Engine builder (Turso/libSQL first)
# -----------------------------

def build_engine() -> Tuple[Engine, Dict[str, str]]:
    turso_url = _get_secret("TURSO_DATABASE_URL")
    turso_token = _get_secret("TURSO_AUTH_TOKEN")
    info = {
        "using_remote": False,
        "dialect": "",
        "driver": "",
        "sqlalchemy_url": "",
        "sync_url": turso_url or "",
    }

    if turso_url and turso_token:
        engine = create_engine(
            str(turso_url),
            connect_args={"auth_token": str(turso_token)},
            pool_pre_ping=True,
            pool_recycle=180,
        )
        info.update({"using_remote": True, "sqlalchemy_url": str(turso_url)})
    else:
        if not ALLOW_SQLITE_FALLBACK:
            st.error("Turso credentials missing and SQLite fallback disabled. Set TURSO_* or enable ALLOW_SQLITE_FALLBACK for local dev.")
            st.stop()
        local_path = os.environ.get("LOCAL_SQLITE_PATH", "vendors.db")
        engine = create_engine(f"sqlite:///{local_path}")
        info.update({"using_remote": False, "sqlalchemy_url": f"sqlite:///{local_path}"})

    # Probe dialect/driver
    try:
        info["dialect"] = engine.dialect.name
        info["driver"] = getattr(engine.dialect, "driver", "")
    except Exception:
        pass
    return engine, info

engine, engine_info = build_engine()

if not engine_info.get("using_remote"):
    st.warning("Running on local SQLite fallback (remote DB unavailable or disabled).", icon="⚠️")

# -----------------------------
# Auth gate
# -----------------------------

ADMIN_PASSWORD = _get_secret("ADMIN_PASSWORD")
if ADMIN_PASSWORD:
    if "_admin_ok" not in st.session_state:
        with st.form("admin_login", clear_on_submit=False, enter_to_submit=True):
            st.subheader("Admin Login")
            pw = st.text_input("Password", type="password")
            ok = st.form_submit_button("Enter")
        if ok:
            if pw == str(ADMIN_PASSWORD):
                st.session_state._admin_ok = True
                st.rerun()
            else:
                st.error("Incorrect password.")
                st.stop()
        else:
            st.stop()
else:
    st.info("No admin password set in secrets — gate is OFF.")

# -----------------------------
# Utilities
# -----------------------------

PHONE_DIGITS = re.compile(r"\D+")


def norm_phone(s: str | None) -> str:
    if not s:
        return ""
    digits = PHONE_DIGITS.sub("", s)
    return digits[:10]


def title_case_or_blank(s: str | None) -> str:
    if not s:
        return ""
    try:
        return s.strip().title()
    except Exception:
        return s.strip()


def safe_like(q: str) -> str:
    return f"%{q.lower()}%"


@st.cache_data(ttl=60)
def fetch_df(sql: str, params: Dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(sql_text(sql), conn, params=params)
    return df


def exec_sql(sql: str, params: Dict | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text(sql), params or {})


def vendors_df(where: str = "", params: Dict | None = None) -> pd.DataFrame:
    base = (
        "SELECT id, business_name, category, service, contact_name, phone, address, website, notes, keywords "
        "FROM vendors "
    )
    if where:
        base += f"WHERE {where} "
    base += "ORDER BY lower(business_name)"
    return fetch_df(base, params)


def categories_df() -> pd.DataFrame:
    return fetch_df("SELECT id, name FROM categories ORDER BY lower(name)")


def services_df() -> pd.DataFrame:
    return fetch_df("SELECT id, name FROM services ORDER BY lower(name)")


def category_usage() -> pd.DataFrame:
    return fetch_df(
        """
        SELECT c.name as category, COUNT(v.id) as usage
        FROM categories c
        LEFT JOIN vendors v ON lower(trim(v.category)) = lower(trim(c.name))
        GROUP BY c.name
        ORDER BY lower(c.name)
        """
    )


def service_usage() -> pd.DataFrame:
    return fetch_df(
        """
        SELECT s.name as service, COUNT(v.id) as usage
        FROM services s
        LEFT JOIN vendors v ON lower(trim(v.service)) = lower(trim(s.name))
        GROUP BY s.name
        ORDER BY lower(s.name)
        """
    )


def orphan_values(col: str, ref_table: str) -> List[str]:
    # Values used in vendors.<col> but missing from ref table
    q = f"""
        WITH used AS (
          SELECT DISTINCT lower(trim({col})) AS val FROM vendors WHERE {col} IS NOT NULL AND trim({col}) <> ''
        ), ref AS (
          SELECT DISTINCT lower(trim(name)) AS val FROM {ref_table}
        )
        SELECT u.val FROM used u
        LEFT JOIN ref r ON u.val = r.val
        WHERE r.val IS NULL
        ORDER BY u.val
    """
    out = fetch_df(q)["val"].tolist()
    return out


def render_html_table(df: pd.DataFrame, sticky_first_col: bool = False) -> None:
    if df.empty:
        st.info("No rows match your filter.")
        return

    # widths
    cols = list(df.columns)
    widths = []
    for c in cols:
        px = RAW_COL_WIDTHS.get(c, 120)
        widths.append(px)

    # Build HTML
    ths = []
    for i, c in enumerate(cols):
        style = f"min-width:{widths[i]}px;max-width:{widths[i]}px;"
        extra = " sticky-first" if sticky_first_col and i == 0 else ""
        ths.append(f'<th class="wrap{extra}" style="{style}">{html.escape(str(c))}</th>')

    trs = []
    for _, row in df.iterrows():
        tds = []
        for i, c in enumerate(cols):
            val = row[c]
            txt = "" if pd.isna(val) else str(val)
            if c == "website" and txt:
                safe = html.escape(txt)
                # ensure scheme
                href = safe if safe.startswith("http") else f"https://{safe}"
                inner = f'<a href="{href}" target="_blank" rel="noopener">{safe}</a>'
            else:
                inner = html.escape(txt)
            style = f"min-width:{widths[i]}px;max-width:{widths[i]}px;"
            extra = " sticky-first" if sticky_first_col and i == 0 else ""
            tds.append(f'<td class="wrap{extra}" style="{style}">{inner}</td>')
        trs.append("<tr>" + "".join(tds) + "</tr>")

    html_table = f"<table class='providers-grid'><thead><tr>{''.join(ths)}</tr></thead><tbody>{''.join(trs)}</tbody></table>"
    st.components.v1.html(html_table, height=500, scrolling=True)


# -----------------------------
# UI Sections
# -----------------------------

def ui_browse():
    st.subheader("Browse Providers")

    q = st.text_input("Search (partial match across most fields)", placeholder="e.g., plumb or 210-555-…")
    sticky = bool(_get_secret("ADMIN_STICKY_FIRST_COL", False))

    df = vendors_df()

    if q:
        ql = q.lower().strip()
        mask = (
            df["business_name"].str.lower().str.contains(ql, na=False) |
            df["category"].str.lower().str.contains(ql, na=False) |
            df["service"].str.lower().str.contains(ql, na=False) |
            df["contact_name"].str.lower().str.contains(ql, na=False) |
            df["phone"].str.lower().str.contains(ql, na=False) |
            df["address"].str.lower().str.contains(ql, na=False) |
            df["website"].str.lower().str.contains(ql, na=False) |
            df["notes"].str.lower().str.contains(ql, na=False) |
            df["keywords"].str.lower().str.contains(ql, na=False)
        )
        df = df[mask].copy()

    # Hide id? Keep visible in admin; you can toggle via secrets
    if str(_get_secret("ADMIN_HIDE_ID", "0")).lower() in ("1", "true", "yes") and "id" in df.columns:
        df = df.drop(columns=["id"])  # id stays in DB; only hidden in browse

    render_html_table(df, sticky_first_col=sticky)

    # Downloads
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download filtered as providers.csv", df.to_csv(index=False).encode("utf-8"), "providers.csv", "text/csv")
    with c2:
        full = vendors_df()
        st.download_button("Download FULL export (all rows)", full.to_csv(index=False).encode("utf-8"), "providers_full.csv", "text/csv")


def _vendor_form_defaults() -> Dict:
    return {
        "id": None,
        "business_name": "",
        "category": "",
        "service": "",
        "contact_name": "",
        "phone": "",
        "address": "",
        "website": "",
        "notes": "",
        "keywords": "",
    }


def ui_vendors():
    st.subheader("Add / Edit / Delete Provider")

    # Left: selector; Right: form
    left, right = st.columns([2, 3])

    with left:
        df = vendors_df()
        choices = ["<New>"] + [f"{r['id']}: {r['business_name']}" for _, r in df.iterrows()]
        sel = st.selectbox("Select a provider to edit", options=choices, index=0)
        if sel != "<New>":
            vid = int(sel.split(":", 1)[0])
            row = df[df.id == vid].iloc[0].to_dict()
        else:
            vid = None
            row = _vendor_form_defaults()

    with right:
        with st.form("vendor_form", clear_on_submit=False):
            business_name = st.text_input("Provider (Business Name) *", value=row.get("business_name", ""))
            category = st.text_input("Category *", value=row.get("category", ""))
            service = st.text_input("Service (optional)", value=row.get("service", ""))
            c1, c2 = st.columns(2)
            with c1:
                contact_name = st.text_input("Contact Name", value=row.get("contact_name", ""))
                phone = st.text_input("Phone (digits only or formatted)", value=row.get("phone", ""))
                website = st.text_input("Website", value=row.get("website", ""))
            with c2:
                address = st.text_area("Address", value=row.get("address", ""), height=80)
                notes = st.text_area("Notes", value=row.get("notes", ""), height=80)
                keywords = st.text_input("Keywords", value=row.get("keywords", ""))

            save_btn = st.form_submit_button("Save / Update")

        if save_btn:
            # Validation
            errs = []
            if not business_name.strip():
                errs.append("Business Name is required.")
            if not category.strip():
                errs.append("Category is required.")
            phone_norm = norm_phone(phone)
            if phone and len(phone_norm) not in (0, 10):
                errs.append("Phone must have 10 digits (US) or be left blank.")

            if errs:
                for e in errs:
                    st.error(e)
                st.stop()

            # Normalize fields
            bn = business_name.strip()
            cat = category.strip()
            svc = service.strip()
            contact = contact_name.strip()
            addr = address.strip()
            web = website.strip()
            nts = notes.strip()
            kws = keywords.strip()

            if vid is None:
                sql = (
                    "INSERT INTO vendors (business_name, category, service, contact_name, phone, address, website, notes, keywords, updated_by) "
                    "VALUES (:bn, :cat, :svc, :contact, :phone, :addr, :web, :nts, :kws, 'admin')"
                )
                exec_sql(sql, {
                    "bn": bn, "cat": cat, "svc": svc or None, "contact": contact or None,
                    "phone": phone_norm or None, "addr": addr or None, "web": web or None,
                    "nts": nts or None, "kws": kws or None,
                })
                st.success("Added new provider.")
            else:
                sql = (
                    "UPDATE vendors SET business_name=:bn, category=:cat, service=:svc, contact_name=:contact, "
                    "phone=:phone, address=:addr, website=:web, notes=:nts, keywords=:kws, updated_by='admin', updated_at=CURRENT_TIMESTAMP "
                    "WHERE id=:vid"
                )
                exec_sql(sql, {
                    "vid": vid,
                    "bn": bn, "cat": cat, "svc": svc or None, "contact": contact or None,
                    "phone": phone_norm or None, "addr": addr or None, "web": web or None,
                    "nts": nts or None, "kws": kws or None,
                })
                st.success("Saved changes.")
            st.rerun()

        # Delete
        if vid is not None:
            with st.expander("Danger Zone — Delete this provider", expanded=False):
                st.warning("Deleting a provider is permanent.")
                if st.button("Delete Provider", type="primary"):
                    exec_sql("DELETE FROM vendors WHERE id=:vid", {"vid": vid})
                    st.success("Deleted.")
                    st.rerun()


def _edit_block_header(label: str, usage_df: pd.DataFrame, orphans: List[str]):
    st.markdown(f"### {label} Admin")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("Usage counts (based on vendors table):")
        st.dataframe(usage_df, use_container_width=True, hide_index=True)
    with c2:
        if orphans:
            st.warning("Orphans in vendors (not in reference table):\n\n" + "\n".join(orphans))
        else:
            st.info("No orphan values detected.")


def ui_categories():
    cats = categories_df()
    usage = category_usage()
    orph = orphan_values("category", "categories")

    _edit_block_header("Categories", usage, orph)

    st.markdown("**Add Category**")
    new_name = st.text_input("New category name")
    if st.button("Add Category"):
        if not new_name.strip():
            st.error("Name required.")
        else:
            exec_sql("INSERT INTO categories(name) VALUES(:n)", {"n": new_name.strip()})
            st.success("Added category.")
            st.rerun()

    st.info(
        "**How to edit Categories:** Pick a category. Preview usage. Choose to (a) Reassign vendors to another category (existing or new) and delete old, (b) Rename the category, or (c) Delete if usage is 0.",
    )

    names = cats["name"].tolist()
    if not names:
        st.info("No categories yet.")
        return

    sel = st.selectbox("Choose a category to modify", options=names)

    # Reassign vendors (then delete old)
    st.markdown("**Reassign vendors (optional) and delete old category**")
    c1, c2 = st.columns(2)
    with c1:
        reassign_to = st.text_input("Reassign to (existing or new)")
    with c2:
        do_reassign = st.button("Reassign vendors to above")

    if do_reassign:
        if not reassign_to.strip():
            st.error("Provide a target category name.")
        else:
            exec_sql("UPDATE vendors SET category=:to WHERE lower(trim(category))=lower(trim(:frm))", {"to": reassign_to.strip(), "frm": sel})
            # ensure target exists in categories
            exists = fetch_df("SELECT 1 FROM categories WHERE lower(trim(name))=lower(trim(:n))", {"n": reassign_to.strip()})
            if exists.empty:
                exec_sql("INSERT INTO categories(name) VALUES(:n)", {"n": reassign_to.strip()})
            st.success("Reassigned vendors.")
            st.rerun()

    st.markdown("**Rename category**")
    new_cat = st.text_input("New name", key="rename_cat")
    if st.button("Rename"):
        if not new_cat.strip():
            st.error("Provide a new name.")
        else:
            exec_sql("UPDATE categories SET name=:to WHERE lower(trim(name))=lower(trim(:frm))", {"to": new_cat.strip(), "frm": sel})
            exec_sql("UPDATE vendors SET category=:to WHERE lower(trim(category))=lower(trim(:frm))", {"to": new_cat.strip(), "frm": sel})
            st.success("Renamed category and updated vendors.")
            st.rerun()

    st.markdown("**Delete category** (usage must be 0)")
    if st.button("Delete Category"):
        used = fetch_df("SELECT COUNT(*) as n FROM vendors WHERE lower(trim(category))=lower(trim(:n))", {"n": sel}).iloc[0]["n"]
        if used:
            st.error("Cannot delete: category still in use. Reassign vendors first.")
        else:
            exec_sql("DELETE FROM categories WHERE lower(trim(name))=lower(trim(:n))", {"n": sel})
            st.success("Deleted category.")
            st.rerun()


def ui_services():
    svcs = services_df()
    usage = service_usage()
    orph = orphan_values("service", "services")

    _edit_block_header("Services", usage, orph)

    st.markdown("**Add Service**")
    new_name = st.text_input("New service name")
    if st.button("Add Service"):
        if not new_name.strip():
            st.error("Name required.")
        else:
            exec_sql("INSERT INTO services(name) VALUES(:n)", {"n": new_name.strip()})
            st.success("Added service.")
            st.rerun()

    names = svcs["name"].tolist()
    if not names:
        st.info("No services yet.")
        return

    sel = st.selectbox("Choose a service to modify", options=names)

    st.markdown("**Rename service**")
    new_svc = st.text_input("New name", key="rename_svc")
    if st.button("Rename Service"):
        if not new_svc.strip():
            st.error("Provide a new name.")
        else:
            exec_sql("UPDATE services SET name=:to WHERE lower(trim(name))=lower(trim(:frm))", {"to": new_svc.strip(), "frm": sel})
            exec_sql("UPDATE vendors SET service=:to WHERE lower(trim(service))=lower(trim(:frm))", {"to": new_svc.strip(), "frm": sel})
            st.success("Renamed service and updated vendors.")
            st.rerun()

    st.markdown("**Delete service** (usage must be 0)")
    if st.button("Delete Service"):
        used = fetch_df("SELECT COUNT(*) as n FROM vendors WHERE lower(trim(service))=lower(trim(:n))", {"n": sel}).iloc[0]["n"]
        if used:
            st.error("Cannot delete: service still in use.")
        else:
            exec_sql("DELETE FROM services WHERE lower(trim(name))=lower(trim(:n))", {"n": sel})
            st.success("Deleted service.")
            st.rerun()


def ui_maintenance():
    st.subheader("Maintenance")

    st.markdown("**Bulk fixes (irreversible)**")
    st.warning("These actions modify many rows. Consider exporting a CSV backup first.")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Title-case business & contact names"):
            df = vendors_df()
            for _, r in df.iterrows():
                exec_sql(
                    "UPDATE vendors SET business_name=:bn, contact_name=:cn, updated_by='admin' WHERE id=:id",
                    {"bn": title_case_or_blank(r["business_name"]), "cn": title_case_or_blank(r.get("contact_name")), "id": int(r["id"])},
                )
            st.success("Done.")
    with c2:
        if st.button("Normalize phones to 10 digits"):
            df = vendors_df()
            for _, r in df.iterrows():
                exec_sql(
                    "UPDATE vendors SET phone=:p, updated_by='admin' WHERE id=:id",
                    {"p": (norm_phone(r.get("phone")) or None), "id": int(r["id"])},
                )
            st.success("Done.")
    with c3:
        if st.button("Backfill updated_at for all rows"):
            exec_sql("UPDATE vendors SET updated_at=CURRENT_TIMESTAMP, updated_by=COALESCE(updated_by,'admin')")
            st.success("Done.")

    st.markdown("**Exports**")
    df_all = vendors_df()
    st.download_button("Download full providers.csv", df_all.to_csv(index=False).encode("utf-8"), "providers.csv", "text/csv")


def ui_debug():
    if st.button("Show Debug / Status", key="dbg_btn"):
        st.write("Status & Secrets (debug)")
        st.json({
            "using_remote": engine_info.get("using_remote"),
            "strategy": "embedded_replica" if engine_info.get("driver") == "libsql" else "sqlite",
            "sqlalchemy_url": engine_info.get("sqlalchemy_url"),
            "dialect": engine_info.get("dialect"),
            "driver": engine_info.get("driver"),
            "sync_url": engine_info.get("sync_url"),
        })
        # Quick DB probe
        try:
            probe = {
                "vendors_columns": fetch_df("PRAGMA table_info(vendors)")["name"].tolist(),
                "categories_columns": fetch_df("PRAGMA table_info(categories)")["name"].tolist(),
                "services_columns": fetch_df("PRAGMA table_info(services)")["name"].tolist(),
            }
            cnts = fetch_df(
                "SELECT (SELECT COUNT(1) FROM vendors) AS vendors, (SELECT COUNT(1) FROM categories) AS categories, (SELECT COUNT(1) FROM services) AS services"
            ).iloc[0].to_dict()
            probe["counts"] = cnts
            st.json(probe)
        except Exception as e:
            st.error(f"Probe failed: {e}")


# -----------------------------
# Main
# -----------------------------

def main():
    # Optional header help
    if ADMIN_HELP_MD:
        with st.expander(ADMIN_HELP_TITLE or "Admin Tips", expanded=False):
            st.markdown(ADMIN_HELP_MD)

    tabs = st.tabs(["Browse", "Vendors", "Categories", "Services", "Maintenance", "Debug"])
    with tabs[0]:
        ui_browse()
    with tabs[1]:
        ui_vendors()
    with tabs[2]:
        ui_categories()
    with tabs[3]:
        ui_services()
    with tabs[4]:
        ui_maintenance()
    with tabs[5]:
        ui_debug()


if __name__ == "__main__":
    main()
