# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# ---- register libsql dialect (must be AFTER "import streamlit as st") ----
try:
    import sqlalchemy_libsql  # noqa: F401
except Exception:
    pass
# ---- end dialect registration ----


# -----------------------------
# Helpers
# -----------------------------
def _as_bool(v, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


def _get_secret(name: str, default: Optional[Any] = None) -> Optional[Any]:
    try:
        return st.secrets.get(name, default)  # type: ignore[attr-defined]
    except Exception:
        return default


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _digits_only(s: str) -> str:
    return re.sub(r"\D+", "", s or "")


def _fmt_phone(s: str) -> str:
    d = _digits_only(s)
    if len(d) == 10:
        return f"({d[0:3]}) {d[3:6]}-{d[6:10]}"
    return s or ""


# ---- service synonyms / keyword expander ----
_SERVICE_SYNONYMS = {
    "insurance agent": {"insurance", "broker", "policy", "auto", "home"},
    "electrician": {"electrical", "wiring", "breaker"},
    "plumber": {"plumbing", "leak", "drain", "pipe"},
    "roofer": {"roof", "shingle", "leak"},
    "landscaper": {"landscape", "yard", "lawn", "xeriscape"},
}


def _ckw(category: str, service: str, business_name: str, extra_keywords: str = "") -> str:
    """
    Build computed_keywords: lowercased, deduped keyword bag from cat/svc/name (+ synonyms) + explicit keywords.
    """
    parts: List[str] = []
    for v in (category, service, business_name, extra_keywords):
        if v:
            parts.extend(re.split(r"[,\s;/]+", str(v).strip().lower()))
    parts = [p for p in parts if p]
    svc = (service or "").strip().lower()
    parts.extend(list(_SERVICE_SYNONYMS.get(svc, set())))
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return " ".join(out)


def _make_search_blob(row: dict) -> str:
    """
    Build a normalized search blob for the Browse global search.
    Merge category/service/business_name/contact/address/website/notes/keywords/computed_keywords.
    """
    fields = [
        "business_name", "category", "service", "contact_name",
        "address", "website", "notes", "keywords", "computed_keywords",
    ]
    tokens: List[str] = []
    for f in fields:
        v = (row.get(f) or "").strip()
        if v:
            tokens.append(v)
    blob = " ".join(tokens).lower()
    blob = re.sub(r"\s+", " ", blob)
    return blob


# -----------------------------
# Engine / DB bootstrap
# -----------------------------
def build_engine() -> Tuple[Engine, Dict[str, Any]]:
    """
    Prefer embedded replica (sqlite+libsql) with sync_url (Turso) if secrets present,
    else fallback to local sqlite file.
    """
    info: Dict[str, Any] = {}
    turso_url = _get_secret("TURSO_DATABASE_URL", "")
    turso_token = _get_secret("TURSO_AUTH_TOKEN", "")
    embedded_db_path = _get_secret("EMBEDDED_DB_PATH", "vendors-embedded.db")

    using_remote = False
    try:
        if turso_url and embedded_db_path:
            db_file = embedded_db_path if str(embedded_db_path).startswith("/") else f"/mount/src/{embedded_db_path}"
            db_url = f"sqlite+libsql:///{db_file}"
            connect_args = {"sync_url": turso_url}
            if turso_token:
                connect_args["auth_token"] = turso_token
            engine = create_engine(db_url, connect_args=connect_args, pool_pre_ping=True, future=True)
            using_remote = True
            info.update({
                "strategy": "embedded_replica",
                "sqlalchemy_url": db_url,
                "sync_url": turso_url,
            })
        else:
            db_file = embedded_db_path if str(embedded_db_path).startswith("/") else f"/mount/src/{embedded_db_path}"
            db_url = f"sqlite:///{db_file}"
            engine = create_engine(db_url, pool_pre_ping=True, future=True)
            info.update({"strategy": "local_sqlite", "sqlalchemy_url": db_url})
        info["dialect"] = "sqlite"
        info["driver"] = "libsql" if "embedded_replica" in info.get("strategy", "") else "pysqlite"
        info["using_remote"] = using_remote
        return engine, info
    except Exception as e:
        # final fallback: temp in-memory (keeps app alive for diagnostics)
        engine = create_engine("sqlite://", future=True)
        info.update({"strategy": "memory_fallback", "error": str(e), "using_remote": False})
        return engine, info


def ensure_schema(engine: Engine) -> None:
    """
    Create tables if not present; indexes created idempotently.
    """
    stmts: List[str] = [
        """
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS services (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
        """,
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
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT,
            updated_by TEXT,
            computed_keywords TEXT
        )
        """,
    ]
    idx_stmts: List[str] = [
        "CREATE INDEX IF NOT EXISTS idx_vendors_ckw ON vendors(computed_keywords)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_phone ON vendors(phone)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_svc_lower ON vendors(LOWER(service))",
    ]
    with engine.begin() as conn:
        for s in stmts:
            conn.execute(sql_text(s))
        for s in idx_stmts:
            conn.execute(sql_text(s))


def drop_redundant_plain_indexes(engine: Engine) -> None:
    """
    Drop legacy plain (case-sensitive) indexes we don't use:
      - idx_vendors_bus  (business_name)
      - idx_vendors_cat  (category)
    """
    with engine.begin() as conn:
        conn.execute(sql_text("DROP INDEX IF EXISTS idx_vendors_bus"))
        conn.execute(sql_text("DROP INDEX IF EXISTS idx_vendors_cat"))


def _row_to_dict(row) -> dict:
    return {k: row[k] for k in row.keys()}


def fetch_categories(engine: Engine) -> List[str]:
    with engine.begin() as conn:
        res = conn.execute(sql_text("SELECT name FROM categories ORDER BY name COLLATE NOCASE"))
        return [r[0] for r in res.fetchall()]


def fetch_services(engine: Engine) -> List[str]:
    with engine.begin() as conn:
        res = conn.execute(sql_text("SELECT name FROM services ORDER BY name COLLATE NOCASE"))
        return [r[0] for r in res.fetchall()]


def list_vendors(engine: Engine) -> List[dict]:
    with engine.begin() as conn:
        res = conn.execute(sql_text("""
            SELECT id, category, service, business_name, contact_name, phone, address,
                   website, notes, keywords, created_at, updated_at, updated_by, computed_keywords
            FROM vendors
            ORDER BY business_name COLLATE NOCASE
        """))
        return [_row_to_dict(r) for r in res.mappings().all()]


def insert_vendor(engine: Engine, row: dict, updated_by: str = "admin") -> int:
    row = {**row}
    row["phone"] = _digits_only(row.get("phone", ""))
    row["computed_keywords"] = _ckw(row.get("category", ""), row.get("service", ""), row.get("business_name", ""), row.get("keywords", ""))
    row["created_at"] = _now()
    row["updated_at"] = _now()
    row["updated_by"] = updated_by
    with engine.begin() as conn:
        res = conn.execute(sql_text("""
            INSERT INTO vendors (category, service, business_name, contact_name, phone, address, website, notes, keywords,
                                 created_at, updated_at, updated_by, computed_keywords)
            VALUES (:category, :service, :business_name, :contact_name, :phone, :address, :website, :notes, :keywords,
                    :created_at, :updated_at, :updated_by, :computed_keywords)
        """), row)
        return int(res.lastrowid)


def update_vendor(engine: Engine, vid: int, row: dict, updated_by: str = "admin") -> None:
    row = {**row}
    row["phone"] = _digits_only(row.get("phone", ""))
    row["computed_keywords"] = _ckw(row.get("category", ""), row.get("service", ""), row.get("business_name", ""), row.get("keywords", ""))
    row["updated_at"] = _now()
    row["updated_by"] = updated_by
    row["id"] = vid
    with engine.begin() as conn:
        conn.execute(sql_text("""
            UPDATE vendors
            SET category=:category, service=:service, business_name=:business_name, contact_name=:contact_name,
                phone=:phone, address=:address, website=:website, notes=:notes, keywords=:keywords,
                updated_at=:updated_at, updated_by=:updated_by, computed_keywords=:computed_keywords
            WHERE id=:id
        """), row)


def delete_vendor(engine: Engine, vid: int) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("DELETE FROM vendors WHERE id=:id"), {"id": vid})


def backfill_computed_keywords(engine: Engine) -> int:
    """
    Recompute computed_keywords where NULL or blank (idempotent).
    """
    with engine.begin() as conn:
        rows = conn.execute(sql_text("""
            SELECT id, category, service, business_name, keywords
            FROM vendors
            WHERE computed_keywords IS NULL OR TRIM(computed_keywords)=''
        """)).mappings().all()
        count = 0
        for r in rows:
            ck = _ckw(r["category"] or "", r["service"] or "", r["business_name"] or "", r["keywords"] or "")
            conn.execute(sql_text("""
                UPDATE vendors SET computed_keywords=:ck, updated_at=:ts WHERE id=:id
            """), {"ck": ck, "id": r["id"], "ts": _now()})
            count += 1
        return count


# -----------------------------
# Page config substitutes (title + width without set_page_config)
# -----------------------------
page_title = _get_secret("page_title", "HCR Providers — Admin") or "HCR Providers — Admin"
st.markdown(f"<script>document.title = {json.dumps(page_title)};</script>", unsafe_allow_html=True)

page_left_padding_px = _get_secret("page_left_padding_px", "12")
page_max_width_px = _get_secret("page_max_width_px", "2300")
st.markdown(
    f"""
    <style>
    .block-container {{
        padding-left: {page_left_padding_px}px !important;
        max-width: {page_max_width_px}px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

engine, engine_info = build_engine()
ensure_schema(engine)
drop_redundant_plain_indexes(engine)

# ---- automatic, idempotent backfill on startup ----
RUN_STARTUP_BACKFILL = _as_bool(_get_secret("RUN_STARTUP_BACKFILL", True), default=True)
if RUN_STARTUP_BACKFILL:
    try:
        n_backfilled = backfill_computed_keywords(engine)
        if n_backfilled:
            st.caption(f"Startup backfill: computed_keywords filled for {n_backfilled} rows.")
    except Exception as e:
        st.warning(f"Startup backfill skipped due to error: {e}")

# Session defaults
if "edit_id" not in st.session_state:
    st.session_state["edit_id"] = None


# -----------------------------
# Browse (with Global Search)
# -----------------------------
def page_browse(engine: Engine):
    st.subheader("Browse Providers")

    data = list_vendors(engine)
    if not data:
        st.info("No providers found yet.")
        return

    # Global search box
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        with col1:
            q = st.text_input("Global search (matches any word across all fields)", key="browse_query", placeholder="e.g., plumber, roof, Stone Oak")
        with col2:
            do_search = st.button("Search", use_container_width=True)
        with col3:
            reset = st.button("Reset", use_container_width=True)
    if reset:
        st.session_state["browse_query"] = ""
        q = ""

    # Build blobs once
    for row in data:
        row["_search_blob"] = _make_search_blob(row)

    filtered = data
    if do_search and (q or "").strip():
        q_norm = (q or "").strip().lower()
        terms = [t for t in re.split(r"[,\s;/]+", q_norm) if t]

        def _match(blob: str) -> bool:
            return all(t in blob for t in terms)

        filtered = [r for r in data if _match(r["_search_blob"])]

    st.caption(f"{len(filtered)} of {len(data)} rows")

    def _render_row(r: dict) -> dict:
        out = dict(r)
        out["phone"] = _fmt_phone(out.get("phone", ""))
        return out

    disp_cols = ["id", "business_name", "category", "service", "contact_name", "phone", "address", "website", "notes"]
    df = pd.DataFrame([_render_row(r) for r in filtered], columns=disp_cols)
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Show raw row JSON (debug)"):
        st.code(json.dumps(filtered[:5], indent=2))


# -----------------------------
# Add / Edit / Delete (split labels + bottom delete section)
# -----------------------------
def page_add_edit(engine: Engine):
    st.subheader("Manage Providers")

    categories = fetch_categories(engine)
    services = fetch_services(engine)
    vendors = list_vendors(engine)

    col_add, col_edit = st.columns(2)

    # ---- ADD ----
    with col_add:
        st.markdown("### Add Provider")
        with st.form("form_add", clear_on_submit=True):
            st.markdown("<div style='height: 2px'></div>", unsafe_allow_html=True)  # spacing
            add_business_name = st.text_input("Provider *", key="add_business_name")
            add_category = st.selectbox("Category *", options=[""] + categories, index=0, key="add_category", placeholder="Select category")
            add_service = st.selectbox("Service (optional)", options=[""] + services, index=0, key="add_service", placeholder="Select service")
            add_contact_name = st.text_input("Contact Name", key="add_contact_name")
            add_phone = st.text_input("Phone (10 digits or blank)", key="add_phone")
            add_address = st.text_area("Address", height=80, key="add_address")
            add_website = st.text_input("Website (https://…)", key="add_website")
            add_notes = st.text_area("Notes", height=100, key="add_notes")
            add_keywords = st.text_input("Keywords (comma separated)", key="add_keywords")

            add_submit = st.form_submit_button("Add Provider", type="primary")
            if add_submit:
                errs = []
                if not (add_business_name or "").strip():
                    errs.append("Provider name is required.")
                if not (add_category or "").strip():
                    errs.append("Category is required.")
                d = _digits_only(add_phone)
                if d and len(d) != 10:
                    errs.append("Phone must be 10 digits (or blank).")
                if add_service and re.search(r"[,/;]", add_service):
                    errs.append("Service name must be a single service (no commas or slashes).")

                if errs:
                    st.error(" • " + "\n • ".join(errs))
                else:
                    row = dict(
                        category=add_category.strip(),
                        service=(add_service or "").strip() or None,
                        business_name=add_business_name.strip(),
                        contact_name=(add_contact_name or "").strip() or None,
                        phone=d,
                        address=(add_address or "").strip() or None,
                        website=(add_website or "").strip() or None,
                        notes=(add_notes or "").strip() or None,
                        keywords=(add_keywords or "").strip() or None,
                    )
                    vid = insert_vendor(engine, row)
                    st.success(f"Added provider id={vid}.")
                    st.rerun()

    # ---- EDIT ----
    with col_edit:
        st.markdown("### Edit Provider")
        _edit_options = [f"{v['id']:>4} — {v['business_name']}" for v in vendors]
        with st.form("form_edit"):
            edit_sel = st.selectbox("Select provider to edit", options=[""] + _edit_options, index=0, key="edit_selector")
            sel_row = None
            if edit_sel and edit_sel.strip():
                try:
                    sel_id = int(edit_sel.split("—", 1)[0])
                    sel_row = next((v for v in vendors if v["id"] == sel_id), None)
                except Exception:
                    sel_row = None

            if sel_row:
                _edit_cat_options = sorted(set(categories + ([sel_row["category"]] if sel_row.get("category") else [])))
                _edit_svc_options = sorted(set(services + ([sel_row["service"]] if sel_row.get("service") else [])))

                st.text_input("Provider *", key="edit_business_name", value=sel_row.get("business_name") or "")
                st.selectbox("Category *", options=_edit_cat_options, key="edit_category", placeholder="Select category")
                st.selectbox("Service (optional)", options=[""] + _edit_svc_options, key="edit_service")
                st.text_input("Contact Name", key="edit_contact_name", value=sel_row.get("contact_name") or "")
                st.text_input("Phone (10 digits or blank)", key="edit_phone", value=_fmt_phone(sel_row.get("phone") or ""))
                st.text_area("Address", height=80, key="edit_address", value=sel_row.get("address") or "")
                st.text_input("Website (https://…)", key="edit_website", value=sel_row.get("website") or "")
                st.text_area("Notes", height=100, key="edit_notes", value=sel_row.get("notes") or "")
                st.text_input("Keywords (comma separated)", key="edit_keywords", value=sel_row.get("keywords") or "")

                edit_submit = st.form_submit_button("Save Changes", type="primary")
                if edit_submit:
                    errs = []
                    ebn = (st.session_state.get("edit_business_name") or "").strip()
                    ecat = (st.session_state.get("edit_category") or "").strip()
                    esvc = (st.session_state.get("edit_service") or "").strip()
                    ecn = (st.session_state.get("edit_contact_name") or "").strip()
                    eph = (st.session_state.get("edit_phone") or "").strip()
                    eaddr = (st.session_state.get("edit_address") or "").strip()
                    eweb = (st.session_state.get("edit_website") or "").strip()
                    enotes = (st.session_state.get("edit_notes") or "").strip()
                    ekw = (st.session_state.get("edit_keywords") or "").strip()

                    if not ebn:
                        errs.append("Provider name is required.")
                    if not ecat:
                        errs.append("Category is required.")
                    d = _digits_only(eph)
                    if d and len(d) != 10:
                        errs.append("Phone must be 10 digits (or blank).")
                    if esvc and re.search(r"[,/;]", esvc):
                        errs.append("Service name must be a single service (no commas or slashes).")

                    if errs:
                        st.error(" • " + "\n • ".join(errs))
                    else:
                        update_vendor(engine, sel_row["id"], {
                            "category": ecat,
                            "service": esvc or None,
                            "business_name": ebn,
                            "contact_name": ecn or None,
                            "phone": d,
                            "address": eaddr or None,
                            "website": eweb or None,
                            "notes": enotes or None,
                            "keywords": ekw or None,
                        })
                        st.success(f"Updated provider id={sel_row['id']}.")
                        st.rerun()
            else:
                st.info("Select a provider to edit and press 'Save Changes' to apply updates.")

    # ---- DELETE (bottom, its own form to avoid 'Missing Submit Button') ----
    st.markdown("---")
    st.markdown("### Delete Provider")
    del_options = [f"{v['id']:>4} — {v['business_name']}" for v in vendors]
    with st.form("form_delete"):
        del_sel = st.selectbox("Select provider to delete", options=[""] + del_options, index=0, key="delete_selector")
        confirm = st.checkbox("I understand this will permanently delete the provider.", value=False, key="delete_confirm")
        del_submit = st.form_submit_button("Delete Provider", type="secondary")
        if del_submit:
            if not del_sel or not del_sel.strip():
                st.error("Select a provider to delete.")
            elif not confirm:
                st.error("Please confirm deletion.")
            else:
                try:
                    del_id = int(del_sel.split("—", 1)[0])
                except Exception:
                    del_id = None
                if del_id is None:
                    st.error("Invalid selection.")
                else:
                    delete_vendor(engine, del_id)
                    st.success(f"Deleted provider id={del_id}.")
                    st.rerun()


# -----------------------------
# CSV Restore (simple) with header map + id block
# -----------------------------
def _csv_restore_append(engine: Engine):
    st.markdown("### CSV Restore (Append-only) — Validate & Append")

    up = st.file_uploader("Drag & drop CSV here (header row required).", type=["csv"], accept_multiple_files=False)
    dry_run = st.checkbox("Dry run (validate only)", value=True)
    do_it = st.button("Validate & Append", type="primary")

    required = ["business_name", "category"]
    allowed = {"business_name", "category", "service", "contact_name", "phone", "address", "website", "notes", "keywords"}
    forbidden = {"id"}

    # header synonyms (lowercased, punctuation/space tolerant)
    header_map = {
        "business_name": {"business_name", "name", "provider", "company", "vendor", "business"},
        "category": {"category", "cat", "type"},
        "service": {"service", "svc", "services"},
        "contact_name": {"contact_name", "contact", "attn", "person"},
        "phone": {"phone", "tel", "telephone", "mobile", "cell", "phone#", "phone_number"},
        "address": {"address", "addr", "location", "street"},
        "website": {"website", "url", "site", "web", "homepage"},
        "notes": {"notes", "comment", "remarks", "desc", "description"},
        "keywords": {"keywords", "tags", "kw", "search_terms"},
    }

    def _norm(h: str) -> str:
        h = (h or "").strip().lower()
        h = re.sub(r"[^\w\s]", "", h)        # drop punctuation
        h = re.sub(r"\s+", " ", h).strip()   # collapse spaces
        return h

    def _canonicalize_header(raw: str) -> Optional[str]:
        n = _norm(raw)
        if n in forbidden:
            return "__FORBIDDEN__"
        # exact allowed first
        if n in allowed:
            return n
        # try map
        for canon, aliases in header_map.items():
            if n in aliases:
                return canon
        return None  # unknown header; will be ignored

    if do_it and up is not None:
        try:
            df_raw = pd.read_csv(up)
        except Exception as e:
            st.error(f"CSV read error: {e}")
            return

        # Build header mapping from df_raw.columns → canonical
        col_map: Dict[str, Optional[str]] = {str(c): _canonicalize_header(str(c)) for c in df_raw.columns}

        # hard block: forbidden columns present?
        if any(v == "__FORBIDDEN__" for v in col_map.values()):
            bads = [k for k, v in col_map.items() if v == "__FORBIDDEN__"]
            st.error(f"CSV contains forbidden column(s): {', '.join(bads)}. Remove them and try again.")
            return

        # Construct a normalized df with only allowed canon columns; add missing as empty
        df = pd.DataFrame()
        for orig, canon in col_map.items():
            if canon and canon in allowed:
                df[canon] = df_raw[orig].astype("string").fillna("").map(str).map(str.strip)

        for c in allowed:
            if c not in df.columns:
                df[c] = ""

        # validation
        errs: List[str] = []
        for i, r in df.iterrows():
            # required columns non-empty
            for req in required:
                if not (str(r.get(req) or "").strip()):
                    errs.append(f"Row {i+1}: '{req}' is required.")
            # phone
            d = _digits_only(r.get("phone") or "")
            if d and len(d) != 10:
                errs.append(f"Row {i+1}: phone must be 10 digits or blank.")
            # multi-service guard
            svc = str(r.get("service") or "")
            if svc and re.search(r"[,/;]", svc):
                errs.append(f"Row {i+1}: service must be a single value (no commas or slashes).")

        if errs:
            st.error("Validation errors:\n\n" + "\n".join(f"• {e}" for e in errs))
            st.stop()

        st.info(
            "Validation summary\n\n"
            f"- csv_rows: {len(df)}\n"
            f"- insertable_columns: {sorted(list(allowed))}\n"
            f"- rows_planned: {len(df)}\n"
            f"- explicit_id_blocked: true\n"
            f"- header_map: { {k: v for k, v in col_map.items() if v} }"
        )

        if dry_run:
            st.success("Dry run complete. No changes applied.")
            return

        # append
        ins_rows = 0
        with engine.begin() as conn:
            for _, r in df.iterrows():
                d = _digits_only(r.get("phone") or "")
                ins = dict(
                    category=(r.get("category") or "").strip(),
                    service=(r.get("service") or "").strip() or None,
                    business_name=(r.get("business_name") or "").strip(),
                    contact_name=(r.get("contact_name") or "").strip() or None,
                    phone=d,
                    address=(r.get("address") or "").strip() or None,
                    website=(r.get("website") or "").strip() or None,
                    notes=(r.get("notes") or "").strip() or None,
                    keywords=(r.get("keywords") or "").strip() or None,
                    created_at=_now(),
                    updated_at=_now(),
                    updated_by="csv_restore",
                    computed_keywords=_ckw(
                        (r.get("category") or ""),
                        (r.get("service") or ""),
                        (r.get("business_name") or ""),
                        (r.get("keywords") or ""),
                    ),
                )
                conn.execute(sql_text("""
                    INSERT INTO vendors (category, service, business_name, contact_name, phone, address, website, notes, keywords,
                                         created_at, updated_at, updated_by, computed_keywords)
                    VALUES (:category, :service, :business_name, :contact_name, :phone, :address, :website, :notes, :keywords,
                            :created_at, :updated_at, :updated_by, :computed_keywords)
                """), ins)
                ins_rows += 1

        st.success(f"Appended {ins_rows} rows.")
        st.info("Tip: 'Recompute now' below if needed.")


# -----------------------------
# Quick Probes (read-only health checks)
# -----------------------------
def _quick_probes(engine: Engine):
    st.markdown("### Quick Probes (read-only health checks)")

    out: Dict[str, Any] = {}
    with engine.begin() as conn:
        # counts
        out["counts"] = {
            "vendors": int(conn.execute(sql_text("SELECT COUNT(*) FROM vendors")).scalar() or 0),
            "categories": int(conn.execute(sql_text("SELECT COUNT(*) FROM categories")).scalar() or 0),
            "services": int(conn.execute(sql_text("SELECT COUNT(*) FROM services")).scalar() or 0),
        }

        # vendors index list
        idx = conn.execute(sql_text("PRAGMA index_list('vendors')")).mappings().all()
        out["vendors_indexes"] = [dict(r) for r in idx]

        # category/service counts
        res = conn.execute(sql_text("""
            SELECT category, COALESCE(service, '') AS service, COUNT(*) AS cnt
            FROM vendors
            GROUP BY category, COALESCE(service, '')
            ORDER BY category COLLATE NOCASE, service COLLATE NOCASE
        """)).mappings().all()
        out["category_service_counts"] = [dict(r) for r in res]

        # unused services: services not referenced in vendors.service
        res2 = conn.execute(sql_text("SELECT DISTINCT LOWER(COALESCE(service,'')) AS s FROM vendors")).mappings().all()
        svc_used = {r["s"] for r in res2 if r["s"]}
        all_services = {s.lower() for s in fetch_services(engine)}
        unused = sorted(list(all_services - svc_used))
        out["unused_services"] = unused

    st.code(json.dumps(out, indent=2))


# -----------------------------
# Maintenance (Recompute, CSV Restore, Quick Probes, Help)
# -----------------------------
def page_maintenance(engine: Engine):
    st.subheader("Maintenance / Help")

    colA, colB = st.columns([1, 1])
    if colA.button("Recompute now (computed_keywords)"):
        n = backfill_computed_keywords(engine)
        st.success(f"Recomputed {n} rows.")
    with colB.popover("What does this do?"):
        st.write("Rebuilds missing/blank **computed_keywords** from category/service/name/keywords for faster search.")

    _csv_restore_append(engine)
    _quick_probes(engine)

    # Help / Tips (from secrets if provided)
    help_md = _get_secret("HELP_MD", "")
    with st.expander("Provider Help / Tips"):
        if help_md:
            st.markdown(str(help_md), unsafe_allow_html=True)
        else:
            st.markdown("""
            ### How to use this admin
            - Add new providers on the left; Edit existing on the right.
            - Use CSV Restore to append multiple providers safely (no overwrite).
            - Recompute ensures search keywords are up-to-date.
            """)


# -----------------------------
# Diagnostics
# -----------------------------
def page_diag(engine: Engine, info: Dict[str, Any]):
    st.subheader("Diagnostics / Debug")
    st.code(json.dumps(info, indent=2))

    with engine.begin() as conn:
        # Show schema columns quickly
        def _cols(tbl: str) -> List[str]:
            res = conn.execute(sql_text(f"PRAGMA table_info('{tbl}')")).mappings().all()
            return [r["name"] for r in res]

        diag = {
            "vendors_columns": _cols("vendors"),
            "categories_columns": _cols("categories"),
            "services_columns": _cols("services"),
        }

        diag["counts"] = {
            "vendors": int(conn.execute(sql_text("SELECT COUNT(*) FROM vendors")).scalar() or 0),
            "categories": int(conn.execute(sql_text("SELECT COUNT(*) FROM categories")).scalar() or 0),
            "services": int(conn.execute(sql_text("SELECT COUNT(*) FROM services")).scalar() or 0),
        }

        idx = conn.execute(sql_text("PRAGMA index_list('vendors')")).mappings().all()
        diag["vendors_indexes"] = [dict(r) for r in idx]

    st.code(json.dumps(diag, indent=2))


# -----------------------------
# Category / Service Admin (Add, Rename with cascade, Delete if unused)
# -----------------------------
def _rename_cascade(engine: Engine, field: str, old: str, new: str):
    if field not in {"category", "service"}:
        raise ValueError("field must be 'category' or 'service'")
    with engine.begin() as conn:
        conn.execute(sql_text(f"UPDATE vendors SET {field}=:new WHERE {field}=:old"), {"new": new, "old": old})
        tbl = "categories" if field == "category" else "services"
        conn.execute(sql_text(f"UPDATE {tbl} SET name=:new WHERE name=:old"), {"new": new, "old": old})


def _delete_if_unused(engine: Engine, field: str, name: str) -> bool:
    if field not in {"category", "service"}:
        return False
    with engine.begin() as conn:
        cnt = conn.execute(sql_text(f"SELECT COUNT(*) FROM vendors WHERE {field}=:n"), {"n": name}).scalar() or 0
        if cnt:
            return False
        tbl = "categories" if field == "category" else "services"
        conn.execute(sql_text(f"DELETE FROM {tbl} WHERE name=:n"), {"n": name})
        return True


def page_admin_taxonomy(engine: Engine):
    st.subheader("Category / Service Admin")

    tabs = st.tabs(["Categories", "Services"])

    # --- Categories ---
    with tabs[0]:
        cats = fetch_categories(engine)
        st.markdown("**Add Category**")
        with st.form("cat_add"):
            new_cat = st.text_input("New category name", key="new_cat")
            add_cat = st.form_submit_button("Add Category", type="primary")
            if add_cat:
                name = (new_cat or "").strip()
                if not name:
                    st.error("Category name required.")
                elif name.lower() in {c.lower() for c in cats}:
                    st.warning("Category already exists.")
                else:
                    with engine.begin() as conn:
                        conn.execute(sql_text("INSERT INTO categories(name) VALUES(:n)"), {"n": name})
                    st.success(f"Added category '{name}'.")
                    st.rerun()

        st.markdown("---")
        st.markdown("**Rename Category (cascades to vendors)**")
        with st.form("cat_rename"):
            old = st.selectbox("Current category", options=[""] + cats, index=0, key="old_cat")
            new = st.text_input("New category name", key="new_cat2")
            do_rename = st.form_submit_button("Rename")
            if do_rename:
                if not old:
                    st.error("Select a category.")
                else:
                    new_name = (new or "").strip()
                    if not new_name:
                        st.error("New name required.")
                    else:
                        _rename_cascade(engine, "category", old, new_name)
                        st.success(f"Renamed category '{old}' → '{new_name}'.")
                        st.rerun()

        st.markdown("---")
        st.markdown("**Delete Category (only if unused)**")
        with st.form("cat_delete"):
            target = st.selectbox("Category to delete", options=[""] + cats, index=0, key="del_cat")
            do_del = st.form_submit_button("Delete Category")
            if do_del:
                if not target:
                    st.error("Select a category.")
                else:
                    ok = _delete_if_unused(engine, "category", target)
                    if ok:
                        st.success(f"Deleted category '{target}'.")
                        st.rerun()
                    else:
                        st.warning("Category is in use by vendors; reassign or rename first.")

    # --- Services ---
    with tabs[1]:
        svcs = fetch_services(engine)
        st.markdown("**Add Service**")
        with st.form("svc_add"):
            new_svc = st.text_input("New service name", key="new_svc")
            add_svc = st.form_submit_button("Add Service", type="primary")
            if add_svc:
                name = (new_svc or "").strip()
                if not name:
                    st.error("Service name required.")
                elif name.lower() in {s.lower() for s in svcs}:
                    st.warning("Service already exists.")
                else:
                    with engine.begin() as conn:
                        conn.execute(sql_text("INSERT INTO services(name) VALUES(:n)"), {"n": name})
                    st.success(f"Added service '{name}'.")
                    st.rerun()

        st.markdown("---")
        st.markdown("**Rename Service (cascades to vendors)**")
        with st.form("svc_rename"):
            old = st.selectbox("Current service", options=[""] + svcs, index=0, key="old_svc")
            new = st.text_input("New service name", key="new_svc2")
            do_rename = st.form_submit_button("Rename")
            if do_rename:
                if not old:
                    st.error("Select a service.")
                else:
                    new_name = (new or "").strip()
                    if not new_name:
                        st.error("New name required.")
                    else:
                        _rename_cascade(engine, "service", old, new_name)
                        st.success(f"Renamed service '{old}' → '{new_name}'.")
                        st.rerun()

        st.markdown("---")
        st.markdown("**Delete Service (only if unused)**")
        with st.form("svc_delete"):
            target = st.selectbox("Service to delete", options=[""] + svcs, index=0, key="del_svc")
            do_del = st.form_submit_button("Delete Service")
            if do_del:
                if not target:
                    st.error("Select a service.")
                else:
                    ok = _delete_if_unused(engine, "service", target)
                    if ok:
                        st.success(f"Deleted service '{target}'.")
                        st.rerun()
                    else:
                        st.warning("Service is in use by vendors; reassign or rename first.")


# -----------------------------
# Main (top tabs, no sidebar)
# -----------------------------
def _run_main(engine: Engine):
    tabs = st.tabs([
        "Browse Providers",
        "Add / Edit / Delete",
        "Category / Service Admin",
        "Maintenance / Help",
        "Diagnostics / Debug",
    ])

    with tabs[0]:
        page_browse(engine)
    with tabs[1]:
        page_add_edit(engine)
    with tabs[2]:
        page_admin_taxonomy(engine)
    with tabs[3]:
        page_maintenance(engine)
    with tabs[4]:
        page_diag(engine, engine_info)


# Kick it off
_run_main(engine)
