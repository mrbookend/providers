from __future__ import annotations

import os
import re
from datetime import datetime

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
# Page config & CSS
# -----------------------------
PAGE_TITLE = st.secrets.get("page_title", "Vendors Admin") if hasattr(st, "secrets") else "Vendors Admin"
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
      div[data-testid="stDataFrame"] table {{ white-space: nowrap; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Admin sign-in gate (toggleable)
# -----------------------------
# Disable the sign-in by setting in secrets.toml:
#   require_admin_password = false
REQUIRE_ADMIN_PASSWORD = str(st.secrets.get("require_admin_password", "true")).lower() in {"1", "true", "yes", "y"}

if REQUIRE_ADMIN_PASSWORD:
    ADMIN_PASSWORD = (st.secrets.get("ADMIN_PASSWORD") or os.getenv("ADMIN_PASSWORD") or "").strip()
    if not isinstance(ADMIN_PASSWORD, str) or not ADMIN_PASSWORD:
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
else:
    st.caption("Admin password disabled via secrets (require_admin_password = false).")


# -----------------------------
# CSV Restore helpers
# -----------------------------
REQUIRED_VENDOR_COLUMNS = ["business_name", "category"]  # service optional

def _get_table_columns(engine: Engine, table: str) -> list[str]:
    with engine.connect() as conn:
        res = conn.execute(sql_text(f"SELECT * FROM {table} LIMIT 0"))
        return list(res.keys())

def _fetch_existing_ids(engine: Engine, table: str = "vendors") -> set[int]:
    with engine.connect() as conn:
        rows = conn.execute(sql_text(f"SELECT id FROM {table}")).all()
    return {int(r[0]) for r in rows if r[0] is not None}

def _prepare_csv_for_append(
    engine: Engine,
    csv_df: pd.DataFrame,
    *,
    normalize_phone: bool,
    trim_strings: bool,
    treat_missing_id_as_autoincrement: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[str]]:
    """
    Returns: (with_id_df, without_id_df, rejected_existing_ids, insertable_columns)
    DataFrames are already filtered to allowed columns and safe to insert.
    """
    df = csv_df.copy()

    # Trim strings
    if trim_strings:
        for c in df.columns:
            if pd.api.types.is_object_dtype(df[c]):
                df[c] = df[c].astype(str).str.strip()

    # Normalize phone to digits
    if normalize_phone and "phone" in df.columns:
        df["phone"] = df["phone"].astype(str).str.replace(r"\D+", "", regex=True)

    db_cols = _get_table_columns(engine, "vendors")
    insertable_cols = [c for c in df.columns if c in db_cols]

    # Required columns present?
    missing_req = [c for c in REQUIRED_VENDOR_COLUMNS if c not in df.columns]
    if missing_req:
        raise ValueError(f"Missing required column(s) in CSV: {missing_req}")

    # Handle id column
    has_id = "id" in df.columns
    existing_ids = _fetch_existing_ids(engine)

    if has_id:
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
        # Reject rows colliding with existing ids
        mask_conflict = df["id"].notna() & df["id"].astype("Int64").astype("int", errors="ignore").isin(existing_ids)
        rejected_existing_ids = df.loc[mask_conflict, "id"].dropna().astype(int).tolist()
        df_ok = df.loc[~mask_conflict].copy()

        # Split by having id vs. not
        with_id_df = df_ok[df_ok["id"].notna()].copy()
        without_id_df = df_ok[df_ok["id"].isna()].copy() if treat_missing_id_as_autoincrement else pd.DataFrame(columns=df.columns)
    else:
        rejected_existing_ids = []
        with_id_df = pd.DataFrame(columns=df.columns)
        without_id_df = df.copy()

    # Limit to insertable columns and coerce NaN->None for DB
    def _prep_cols(d: pd.DataFrame, drop_id: bool) -> pd.DataFrame:
        cols = [c for c in insertable_cols if (c != "id" if drop_id else True)]
        if not cols:
            return pd.DataFrame(columns=[])
        dd = d[cols].copy()
        for c in cols:
            dd[c] = dd[c].where(pd.notnull(dd[c]), None)
        return dd

    with_id_df = _prep_cols(with_id_df, drop_id=False)
    without_id_df = _prep_cols(without_id_df, drop_id=True)

    # Duplicate ids inside the CSV itself?
    if "id" in csv_df.columns:
        dup_ids = (
            csv_df["id"]
            .pipe(pd.to_numeric, errors="coerce")
            .dropna()
            .astype(int)
            .duplicated(keep=False)
        )
        if dup_ids.any():
            dups = sorted(csv_df.loc[dup_ids, "id"].dropna().astype(int).unique().tolist())
            raise ValueError(f"Duplicate id(s) inside CSV: {dups}")

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


# -----------------------------
# DB build / schema / helpers
# -----------------------------
def build_engine() -> tuple[Engine, dict]:
    """Use Embedded Replica for Turso (syncs to remote), else fallback to local."""
    info: dict = {}

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

    # Embedded replica: local file that syncs to your remote Turso DB
    try:
        raw = url
        if raw.startswith("sqlite+libsql://"):
            host = raw.split("://", 1)[1].split("?", 1)[0]  # drop any ?secure=true
            sync_url = f"libsql://{host}"
        else:
            sync_url = raw  # already libsql://...

        eng = create_engine(
            "sqlite+libsql:///vendors-embedded.db",
            connect_args={
                "auth_token": token,
                "sync_url": sync_url,
            },
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
        allow_local = os.getenv("FORCE_LOCAL") == "1"
        if allow_local:
            eng = create_engine("sqlite:///vendors.db", pool_pre_ping=True)
            info.update({
                "using_remote": False,
                "sqlalchemy_url": "sqlite:///vendors.db",
                "dialect": eng.dialect.name,
                "driver": getattr(eng.dialect, "driver", ""),
            })
            return eng, info

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

    df["notes_short"] = df.get("notes", "").astype(str).str.replace("\n", " ").str.slice(0, 150)
    df["keywords_short"] = df.get("keywords", "").astype(str).str.replace("\n", " ").str.slice(0, 80)
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


# -----------------------------
# UI
# -----------------------------
engine, engine_info = build_engine()
ensure_schema(engine)

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
    st.caption("Search across common fields; sorting is server-side.")
    q = st.text_input(
        "",
        placeholder="Search — e.g., 'plumb' finds partial matches anywhere",
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
    else:
        # Build a Provider (business_name) dropdown to select which vendor to edit
        opts = df_all[["id", "business_name"]].copy()
        opts["business_name"] = opts["business_name"].astype(str).str.strip()

        # Disambiguate duplicates by appending (ID NNN)
        dups = opts["business_name"].duplicated(keep=False)
        opts["label"] = opts["business_name"]
        opts.loc[dups, "label"] = opts["business_name"] + "  (ID " + opts["id"].astype(str) + ")"

        # Sort by label, case-insensitive
        opts = opts.sort_values("label", key=lambda s: s.str.lower())

        labels = opts["label"].tolist()
        id_lookup = dict(zip(labels, opts["id"].tolist()))

        sel_label = st.selectbox(
            "Select provider",
            options=labels,
            index=0 if labels else None,
            placeholder="Type to search a provider name",
        )

        if not sel_label:
            st.info("Select a provider to edit.")
        else:
            sel_id = int(id_lookup[sel_label])
            row_sel = df_all.loc[df_all["id"] == sel_id]
            if row_sel.empty:
                st.warning("Selected provider not found. Try refreshing the page.")
            else:
                row = row_sel.iloc[0]

                # Preselects for category/service
                cat_options = cats if cats else []
                cat_index = (
                    cat_options.index(row["category"])
                    if (row.get("category") in cat_options and cat_options) else None
                )

                svc_options = [""] + servs if servs else [""]
                svc_index = (
                    svc_options.index(row.get("service"))
                    if str(row.get("service")) in svc_options else 0
                )

                with st.form("edit_vendor"):
                    col1, col2 = st.columns(2)
                    with col1:
                        business_name_e = st.text_input("Provider *", row.get("business_name", ""))
                        category_e = st.selectbox("Category *", options=cat_options, index=cat_index)
                        service_e = st.selectbox("Service (optional)", options=svc_options, index=svc_index)
                        contact_name_e = st.text_input("Contact Name", row.get("contact_name", "") or "")
                        phone_e = st.text_input("Phone (10 digits or blank)", row.get("phone", "") or "")
                    with col2:
                        address_e = st.text_area("Address", row.get("address", "") or "", height=80)
                        website_e = st.text_input("Website (https://…)", row.get("website", "") or "")
                        notes_e = st.text_area("Notes", row.get("notes", "") or "", height=100)
                        keywords_e = st.text_input("Keywords (comma separated)", row.get("keywords", "") or "")
                    c1, c2 = st.columns([1, 1])
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
                                "category": (category_e or "").strip(),
                                "service": (service_e or "").strip(),
                                "business_name": (business_name_e or "").strip(),
                                "contact_name": (contact_name_e or "").strip(),
                                "phone": phone_norm,
                                "address": (address_e or "").strip(),
                                "website": url,
                                "notes": (notes_e or "").strip(),
                                "keywords": (keywords_e or "").strip(),
                                "now": now,
                                "user": os.getenv("USER", "admin"),
                                "id": int(sel_id),
                            })
                        st.success("Vendor updated.")
                        st.rerun()

                if delete_btn:
                    with engine.begin() as conn:
                        conn.execute(sql_text("DELETE FROM vendors WHERE id=:id"), {"id": int(sel_id)})
                    st.success("Vendor deleted.")
                    st.rerun()


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

    st.divider()
    st.subheader("Data cleanup")

    # Normalize phones and title-case names (vendors)
    if st.button("Normalize phones (digits only) & title-case business/contacts"):
        with engine.begin() as conn:
            rows = conn.execute(
                sql_text("SELECT id, phone, business_name, contact_name FROM vendors")
            ).fetchall()
            for r in rows:
                pid = int(r[0])
                phone_norm = _normalize_phone(r[1] or "")
                bname = (r[2] or "").strip().title()
                cname = (r[3] or "").strip().title()
                conn.execute(
                    sql_text(
                        "UPDATE vendors SET phone=:p, business_name=:b, contact_name=:c WHERE id=:id"
                    ),
                    {"p": phone_norm, "b": bname, "c": cname, "id": pid},
                )
        st.success("Normalization complete.")

    # NEW: Title-case & de-duplicate Categories/Services everywhere
    if st.button("Title-case categories/services & merge duplicates (safe)"):
        def _titlecase_merge(table: str, col: str):
            with engine.begin() as conn:
                rows = conn.execute(sql_text(f"SELECT id, {col} FROM {table}")).fetchall()
                # rows: [(id, name)] for categories/services tables
                if table in {"vendors"}:
                    raise RuntimeError("Do not call with vendors")

            # Build canonical buckets
            buckets: dict[str, list[tuple[int, str]]] = {}
            for rid, name in rows:
                name_s = (name or "").strip()
                if not name_s:
                    continue
                canon = name_s.title()
                buckets.setdefault(canon, []).append((int(rid), name_s))

            with engine.begin() as conn:
                for canon, entries in buckets.items():
                    # Ensure one primary row exists with the canonical name
                    primary_id = None
                    for rid, name_s in entries:
                        if name_s == canon:
                            primary_id = rid
                            break
                    if primary_id is None:
                        # rename the first entry to canonical
                        primary_id = entries[0][0]
                        conn.execute(sql_text(f"UPDATE {table} SET name=:canon WHERE id=:id"),
                                     {"canon": canon, "id": primary_id})

                    # Update vendors to use canonical value for all old spellings
                    for _, name_s in entries:
                        if table == "categories":
                            conn.execute(sql_text(
                                "UPDATE vendors SET category=:canon WHERE category=:old"
                            ), {"canon": canon, "old": name_s})
                        else:  # services
                            conn.execute(sql_text(
                                "UPDATE vendors SET service=:canon WHERE service=:old"
                            ), {"canon": canon, "old": name_s})

                    # Delete duplicate rows (keep primary)
                    for rid, name_s in entries:
                        if rid != primary_id:
                            conn.execute(sql_text(f"DELETE FROM {table} WHERE id=:id"), {"id": rid})

        _titlecase_merge("categories", "name")
        _titlecase_merge("services", "name")
        st.success("Categories/Services title-cased and duplicates merged across tables.")

    # Backfill timestamps
    if st.button("Backfill created_at/updated_at when missing"):
        now = datetime.utcnow().isoformat(timespec="seconds")
        with engine.begin() as conn:
            conn.execute(
                sql_text(
                    "UPDATE vendors SET created_at=COALESCE(created_at, :now), updated_at=COALESCE(updated_at, :now)"
                ),
                {"now": now},
            )
        st.success("Backfill complete.")

    # Trim extra whitespace across common text fields (preserves newlines in notes)
    if st.button("Trim whitespace in text fields (safe)"):
        changed = 0
        with engine.begin() as conn:
            rows = conn.execute(
                sql_text(
                    """
                    SELECT id, category, service, business_name, contact_name, address, website, notes, keywords, phone
                    FROM vendors
                    """
                )
            ).fetchall()

            def clean_soft(s: str | None) -> str:
                s = (s or "").strip()
                s = re.sub(r"[ \t]+", " ", s)  # collapse spaces/tabs; keep newlines
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
                    "notes":         clean_soft(r[7]),
                    "keywords":      clean_soft(r[8]),
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
    SHOW_DEBUG = bool(st.secrets.get("show_debug", True))
    if SHOW_DEBUG:
        st.subheader("Status & Secrets (debug)")
        st.json(engine_info)

        with engine.begin() as conn:
            vendors_cols = conn.execute(sql_text("PRAGMA table_info(vendors)")).fetchall()
            categories_cols = conn.execute(sql_text("PRAGMA table_info(categories)")).fetchall()
            services_cols = conn.execute(sql_text("PRAGMA table_info(services)")).fetchall()

            # Index presence (vendors)
            idx_rows = conn.execute(sql_text("PRAGMA index_list(vendors)")).fetchall()
            vendors_indexes = [
                {"seq": r[0], "name": r[1], "unique": bool(r[2]), "origin": r[3], "partial": bool(r[4])}
                for r in idx_rows
            ]

            # Null timestamp counts
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
    else:
        st.info("Debug panel disabled via secrets (show_debug = false).")
