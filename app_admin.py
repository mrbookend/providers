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


# ==== BEGIN PATCH 1: ISO timestamp helper ====
def _now() -> str:
    """UTC timestamp in strict ISO format with 'T' separator."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
# ==== END PATCH 1 ====


def _digits_only(s: str) -> str:
    return re.sub(r"\D+", "", s or "")


def _fmt_phone(s: str) -> str:
    d = _digits_only(s)
    if len(d) == 10:
        return f"({d[0:3]}) {d[3:6]}-{d[6:10]}"
    return s or ""


# ==== BEGIN PATCH 2: Ingress normalization + TX canonicalizer ====
def _s(x: Optional[str]) -> str:
    """Normalize text ingress: None -> '', strip whitespace."""
    return (x or "").strip()

_TX_PAT = re.compile(r"(?:^|[\s,])TX(?:[\s,]|$)", re.IGNORECASE)

def _canon_address_tx(addr: Optional[str]) -> str:
    """
    Canonicalize address to include 'TX' if no state is present.
    Conservative: if 'TX' already appears (any case), leave as-is.
    """
    a = _s(addr)
    if not a:
        return ""
    if _TX_PAT.search(a):
        return a
    # Heuristic: add ', TX' if there is at least a street/city token
    if a.endswith(","):
        return a + " TX"
    if re.search(r"\d", a) or "," in a or " " in a:
        return a + ", TX"
    return a  # too ambiguous; leave alone
# ==== END PATCH 2 ====


# ---- service synonyms / keyword expander ----
_SERVICE_SYNONYMS = {
    "insurance agent": {"insurance", "broker", "policy", "auto", "home"},
    "electrician": {"electrical", "wiring", "breaker"},
    "plumber": {"plumbing", "leak", "drain", "pipe"},
    "roofer": {"roof", "shingle", "leak"},
    "landscaper": {"landscape", "yard", "lawn", "xeriscape"},
}

# ==== BEGIN PATCH 3: Multi-service single-label whitelist ====
_SINGLE_SLASH_SERVICE_WHITELIST = {
    "Optometry / Eye Exam",
    "Heating / Cooling",
}
# ==== END PATCH 3 ====


def _ckw(category: str, service: str, business_name: str, extra_keywords: str = "") -> str:
    """Build computed_keywords from cat/svc/name (+ synonyms) + explicit keywords."""
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


# ==== BEGIN PATCH 4: Search blob dedupe + phone + 7to7 normalization ====
def _make_search_blob(row: dict) -> str:
    """
    Concat fields (lowercased) for Browse global search, including computed_keywords.
    Normalize '&'→'and', unify hyphen/dash variants, include phone (digits & formatted),
    and de-duplicate tokens. Normalize any '7 to 7' style to '7to7'.
    """
    fields = [
        "business_name", "category", "service", "contact_name",
        "address", "website", "notes", "keywords", "computed_keywords",
    ]
    parts = [(row.get(f) or "").strip() for f in fields if (row.get(f) or "").strip()]

    # include both raw digits and formatted phone in the blob
    ph_digits = _digits_only(row.get("phone", ""))
    if ph_digits:
        parts.append(ph_digits)
        if len(ph_digits) == 10:
            parts.append(_fmt_phone(ph_digits))

    blob = " ".join(parts).lower()

    # Normalize '&' to 'and'
    blob = blob.replace("&", " and ")

    # Normalize different dashes/hyphens to spaces
    blob = re.sub(r"[\u002D\u2010\u2011\u2012\u2013\u2014\u2212]+", " ", blob)

    # Normalize 7-to-7 variants
    blob = re.sub(r"\b7\s*[-–—]?\s*to\s*[-–—]?\s*7\b", "7to7", blob)

    # Collapse whitespace
    blob = re.sub(r"\s+", " ", blob).strip()

    # De-duplicate tokens, keep order
    tokens = blob.split(" ")
    seen, out = set(), []
    for t in tokens:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return " ".join(out)
# ==== END PATCH 4 ====


def _clear_browse_query():
    """Reset callback for Browse search input (avoids React invariant errors)."""
    st.session_state["browse_query"] = ""


# -----------------------------
# Engine / DB bootstrap
# -----------------------------
def build_engine() -> Tuple[Engine, Dict[str, Any]]:
    """Prefer embedded replica (sqlite+libsql) with Turso sync; fallback to local sqlite file."""
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
        engine = create_engine("sqlite://", future=True)
        info.update({"strategy": "memory_fallback", "error": str(e), "using_remote": False})
        return engine, info


def ensure_schema(engine: Engine) -> None:
    """Create tables if not present; create useful indexes idempotently."""
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
        # legacy case-insensitive helpers may already exist (cat_lower, bus_lower); not recreated here
    ]
    with engine.begin() as conn:
        for s in stmts:
            conn.execute(sql_text(s))
        for s in idx_stmts:
            conn.execute(sql_text(s))


def drop_redundant_plain_indexes(engine: Engine) -> None:
    """Drop legacy plain (case-sensitive) indexes we don't use."""
    with engine.begin() as conn:
        conn.execute(sql_text("DROP INDEX IF EXISTS idx_vendors_bus"))
        conn.execute(sql_text("DROP INDEX IF EXISTS idx_vendors_cat"))


def drop_legacy_keyword_index(engine: Engine) -> None:
    """Drop legacy index on raw 'keywords' (unused by current blob-based search)."""
    with engine.begin() as conn:
        conn.execute(sql_text("DROP INDEX IF EXISTS idx_vendors_kw"))


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


# ==== BEGIN PATCH 2 hooks: sanitize/canonicalize on write ====
def _sanitize_row_for_write(row: dict) -> dict:
    """Apply ingress normalization to all text fields and canonicalize address."""
    out = dict(row)
    for k in ("category", "service", "business_name", "contact_name", "address", "website", "notes", "keywords"):
        out[k] = _s(out.get(k))
    out["address"] = _canon_address_tx(out.get("address", ""))
    # phone already normalized elsewhere to digits-only
    return out
# ==== END PATCH 2 hooks ====


def insert_vendor(engine: Engine, row: dict, updated_by: str = "admin") -> int:
    row = {**row}
    row["phone"] = _digits_only(row.get("phone", ""))
    row = _sanitize_row_for_write(row)  # PATCH 2
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
    row = _sanitize_row_for_write(row)  # PATCH 2
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
    """Recompute computed_keywords where NULL or blank (idempotent)."""
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


# ==== BEGIN PATCH 1 helper: backfill legacy space timestamps to ISO 'T' ====
def backfill_iso_timestamps(engine: Engine) -> Dict[str, int]:
    """
    Convert 'YYYY-MM-DD HH:MM:SS' to 'YYYY-MM-DDTHH:MM:SS' for created_at/updated_at.
    Returns counts changed per column.
    """
    res: Dict[str, int] = {"created_at": 0, "updated_at": 0}
    with engine.begin() as conn:
        # created_at
        res["created_at"] = int(conn.execute(sql_text("""
            WITH to_fix AS (
                SELECT id, created_at FROM vendors
                WHERE created_at IS NOT NULL
                  AND created_at LIKE '% %:%:%'
                  AND created_at NOT LIKE '%T%:%:%'
            )
            UPDATE vendors
               SET created_at = REPLACE(created_at, ' ', 'T')
             WHERE id IN (SELECT id FROM to_fix)
        """)).rowcount or 0)

        # updated_at
        res["updated_at"] = int(conn.execute(sql_text("""
            WITH to_fix AS (
                SELECT id, updated_at FROM vendors
                WHERE updated_at IS NOT NULL
                  AND updated_at LIKE '% %:%:%'
                  AND updated_at NOT LIKE '%T%:%:%'
            )
            UPDATE vendors
               SET updated_at = REPLACE(updated_at, ' ', 'T')
             WHERE id IN (SELECT id FROM to_fix)
        """)).rowcount or 0)
    return res
# ==== END PATCH 1 helper ====


# ==== BEGIN: Page layout CSS (title, width, padding) ====
page_title = _get_secret("page_title", "HCR Providers — Admin") or "HCR Providers — Admin"
st.markdown(f"<script>document.title = {json.dumps(page_title)};</script>", unsafe_allow_html=True)

page_left_padding_px = _get_secret("page_left_padding_px", "12")
page_max_width_px   = _get_secret("page_max_width_px", "2300")
page_top_padding_px = _get_secret("page_top_padding_px", "6")  # << add this secret

st.markdown(
    f"""
    <style>
    .block-container {{
        padding-top: {page_top_padding_px}px !important;   /* << new: tighten top gap */
        padding-left: {page_left_padding_px}px !important;
        max-width: {page_max_width_px}px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
# ==== END: Page layout CSS (title, width, padding) ====


engine, engine_info = build_engine()
ensure_schema(engine)
drop_redundant_plain_indexes(engine)
drop_legacy_keyword_index(engine)

# ---- automatic, idempotent backfill on startup ----
RUN_STARTUP_BACKFILL = _as_bool(_get_secret("RUN_STARTUP_BACKFILL", True), default=True)
if RUN_STARTUP_BACKFILL:
    try:
        n_backfilled = backfill_computed_keywords(engine)
        if n_backfilled:
            st.caption(f"Startup backfill: computed_keywords filled for {n_backfilled} rows.")
    except Exception as e:
        st.warning(f"Startup backfill skipped due to error: {e}")


# -----------------------------
# Browse (with Global Search)
# -----------------------------
def page_browse(engine: Engine):

    data = list_vendors(engine)
    if not data:
        st.info("No providers found yet.")
        return

    # Global search box
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        with col1:
            st.text_input(
                "Global search (matches any word across all fields)",
                key="browse_query",
                placeholder="e.g., plumber, roof, Stone Oak",
            )
        with col2:
            do_search = st.button("Search", use_container_width=True, key="browse_do_search")
        with col3:
            st.button(
                "Reset",
                use_container_width=True,
                key="browse_reset",
                on_click=_clear_browse_query,
            )

    # Build blobs once
    for row in data:
        row["_search_blob"] = _make_search_blob(row)

    filtered = data
    if do_search:
        q_norm = (st.session_state.get("browse_query") or "").strip().lower()
        if q_norm:
            # split on spaces, commas, slashes, ASCII hyphen, and Unicode dashes
            terms = [t for t in re.split(r"[,\s;/\-\u2010-\u2014\u2212]+", q_norm) if t]

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
            st.markdown("<div style='height: 2px'></div>", unsafe_allow_html=True)
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
                # PATCH 3: allow single whitelisted slash label
                if add_service:
                    svc_trim = (add_service or "").strip()
                    if ("/" in svc_trim and svc_trim not in _SINGLE_SLASH_SERVICE_WHITELIST) or re.search(r"[,;]", svc_trim):
                        errs.append("Service must be a single value (no commas; '/' allowed only for approved single-label names).")
                if errs:
                    st.error(" • " + "\n • ".join(errs))
                else:
                    row = dict(
                        category=_s(add_category),
                        service=_s(add_service),
                        business_name=_s(add_business_name),
                        contact_name=_s(add_contact_name),
                        phone=d,
                        address=_canon_address_tx(add_address),
                        website=_s(add_website),
                        notes=_s(add_notes),
                        keywords=_s(add_keywords),
                    )
                    vid = insert_vendor(engine, row)
                    st.success(f"Added provider id={vid}.")
                    st.rerun()

    # ---- EDIT ----
    with col_edit:
        st.markdown("### Edit Provider")

        # Selector outside the form so selection can seed state immediately
        _edit_options = [f"{v['id']:>4} — {v['business_name']}" for v in vendors]
        edit_sel = st.selectbox("Select provider to edit", options=[""] + _edit_options, index=0, key="edit_selector")

        sel_row = None
        sel_id = None
        if edit_sel and edit_sel.strip():
            try:
                sel_id = int(edit_sel.split("—", 1)[0])
                sel_row = next((v for v in vendors if v["id"] == sel_id), None)
            except Exception:
                sel_row = None

        # If selection changed, seed the edit_* fields into session_state
        last_sel = st.session_state.get("edit_last_sel")
        if sel_id and sel_id != last_sel and sel_row:
            st.session_state["edit_last_sel"] = sel_id
            st.session_state["edit_business_name"] = sel_row.get("business_name") or ""
            st.session_state["edit_category"] = sel_row.get("category") or ""
            st.session_state["edit_service"] = sel_row.get("service") or ""
            st.session_state["edit_contact_name"] = sel_row.get("contact_name") or ""
            st.session_state["edit_phone"] = _fmt_phone(sel_row.get("phone") or "")
            st.session_state["edit_address"] = sel_row.get("address") or ""
            st.session_state["edit_website"] = sel_row.get("website") or ""
            st.session_state["edit_notes"] = sel_row.get("notes") or ""
            st.session_state["edit_keywords"] = sel_row.get("keywords") or ""

        # Now render the form that always has a submit button
        with st.form("form_edit", clear_on_submit=False):
            if sel_row:
                _edit_cat_options = sorted(set(categories + ([sel_row["category"]] if sel_row.get("category") else [])))
                _edit_svc_options = sorted(set(services + ([sel_row["service"]] if sel_row.get("service") else [])))
            else:
                _edit_cat_options, _edit_svc_options = ([""], [""])

            # Use session_state-backed widgets (no value=...)
            st.text_input("Provider *", key="edit_business_name", disabled=sel_row is None)
            st.selectbox("Category *", options=_edit_cat_options, key="edit_category", disabled=sel_row is None)
            st.selectbox("Service (optional)", options=[""] + _edit_svc_options if sel_row else [""], key="edit_service", disabled=sel_row is None)
            st.text_input("Contact Name", key="edit_contact_name", disabled=sel_row is None)
            st.text_input("Phone (10 digits or blank)", key="edit_phone", disabled=sel_row is None)
            st.text_area("Address", height=80, key="edit_address", disabled=sel_row is None)
            st.text_input("Website (https://…)", key="edit_website", disabled=sel_row is None)
            st.text_area("Notes", height=100, key="edit_notes", disabled=sel_row is None)
            st.text_input("Keywords (comma separated)", key="edit_keywords", disabled=sel_row is None)

            edit_submit = st.form_submit_button("Save Changes", type="primary", disabled=(sel_row is None))

            if sel_row and edit_submit:
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
                # PATCH 3: allow single whitelisted slash label
                if esvc:
                    if ("/" in esvc and esvc not in _SINGLE_SLASH_SERVICE_WHITELIST) or re.search(r"[,;]", esvc):
                        errs.append("Service must be a single value (no commas; '/' allowed only for approved single-label names).")

                if errs:
                    st.error(" • " + "\n • ".join(errs))
                else:
                    update_vendor(engine, sel_row["id"], {
                        "category": _s(ecat),
                        "service": _s(esvc),
                        "business_name": _s(ebn),
                        "contact_name": _s(ecn),
                        "phone": d,
                        "address": _canon_address_tx(eaddr),
                        "website": _s(eweb),
                        "notes": _s(enotes),
                        "keywords": _s(ekw),
                    })
                    st.success(f"Updated provider id={sel_row['id']}.")
                    st.rerun()

    # ---- DELETE (bottom, its own form) ----
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
        h = re.sub(r"[^\w\s]", "", h)
        h = re.sub(r"\s+", " ", h).strip()
        return h

    def _canonicalize_header(raw: str) -> Optional[str]:
        n = _norm(raw)
        if n in forbidden:
            return "__FORBIDDEN__"
        if n in allowed:
            return n
        for canon, aliases in header_map.items():
            if n in aliases:
                return canon
        return None

    if do_it and up is not None:
        try:
            df_raw = pd.read_csv(up)
        except Exception as e:
            st.error(f"CSV read error: {e}")
            return

        col_map: Dict[str, Optional[str]] = {str(c): _canonicalize_header(str(c)) for c in df_raw.columns}

        if any(v == "__FORBIDDEN__" for v in col_map.values()):
            bads = [k for k, v in col_map.items() if v == "__FORBIDDEN__"]
            st.error(f"CSV contains forbidden column(s): {', '.join(bads)}. Remove them and try again.")
            return

        df = pd.DataFrame()
        for orig, canon in col_map.items():
            if canon and canon in allowed:
                df[canon] = df_raw[orig].astype("string").fillna("").map(str).map(str.strip)
        for c in allowed:
            if c not in df.columns:
                df[c] = ""

        errs: List[str] = []
        for i, r in df.iterrows():
            for req in required:
                if not (str(r.get(req) or "").strip()):
                    errs.append(f"Row {i+1}: '{req}' is required.")
            d = _digits_only(r.get("phone") or "")
            if d and len(d) != 10:
                errs.append(f"Row {i+1}: phone must be 10 digits or blank.")
            svc = str(r.get("service") or "").strip()
            if svc:
                if ("/" in svc and svc not in _SINGLE_SLASH_SERVICE_WHITELIST) or re.search(r"[,;]", svc):
                    errs.append(f"Row {i+1}: service must be a single value (no commas; '/' allowed only for approved single-label names).")

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

        ins_rows = 0
        errors: List[str] = []
        with engine.begin() as conn:
            for idx, r in df.iterrows():
                try:
                    d = _digits_only(r.get("phone") or "")
                    ins = dict(
                        category=_s(r.get("category")),
                        service=_s(r.get("service")),
                        business_name=_s(r.get("business_name")),
                        contact_name=_s(r.get("contact_name")),
                        phone=d,
                        address=_canon_address_tx(r.get("address")),
                        website=_s(r.get("website")),
                        notes=_s(r.get("notes")),
                        keywords=_s(r.get("keywords")),
                        created_at=_now(),
                        updated_at=_now(),
                        updated_by="csv_restore",
                        computed_keywords=_ckw(
                            _s(r.get("category")),
                            _s(r.get("service")),
                            _s(r.get("business_name")),
                            _s(r.get("keywords")),
                        ),
                    )
                    conn.execute(sql_text("""
                        INSERT INTO vendors (category, service, business_name, contact_name, phone, address, website, notes, keywords,
                                             created_at, updated_at, updated_by, computed_keywords)
                        VALUES (:category, :service, :business_name, :contact_name, :phone, :address, :website, :notes, :keywords,
                                :created_at, :updated_at, :updated_by, :computed_keywords)
                    """), ins)
                    ins_rows += 1
                except Exception as ex:
                    errors.append(f"Row {idx+1}: {ex}")

        st.success(f"Appended {ins_rows} rows.")
        if errors:
            st.warning("Some rows failed to append:\n\n" + "\n".join(f"• {e}" for e in errors))
        st.info("Tip: 'Recompute now' below if needed.")


# -----------------------------
# Quick Probes (read-only health checks)
# -----------------------------
def _quick_probes(engine: Engine):
    st.markdown("### Quick Probes (read-only health checks)")

    out: Dict[str, Any] = {}
    with engine.begin() as conn:
        out["counts"] = {
            "vendors": int(conn.execute(sql_text("SELECT COUNT(*) FROM vendors")).scalar() or 0),
            "categories": int(conn.execute(sql_text("SELECT COUNT(*) FROM categories")).scalar() or 0),
            "services": int(conn.execute(sql_text("SELECT COUNT(*) FROM services")).scalar() or 0),
        }
        idx = conn.execute(sql_text("PRAGMA index_list('vendors')")).mappings().all()
        out["vendors_indexes"] = [dict(r) for r in idx]
        res = conn.execute(sql_text("""
            SELECT category, COALESCE(service, '') AS service, COUNT(*) AS cnt
            FROM vendors
            GROUP BY category, COALESCE(service, '')
            ORDER BY category COLLATE NOCASE, service COLLATE NOCASE
        """)).mappings().all()
        out["category_service_counts"] = [dict(r) for r in res]
        res2 = conn.execute(sql_text("SELECT DISTINCT LOWER(COALESCE(service,'')) AS s FROM vendors")).mappings().all()
        svc_used = {r["s"] for r in res2 if r["s"]}
        all_services = {s.lower() for s in fetch_services(engine)}
        out["unused_services"] = sorted(list(all_services - svc_used))

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

    # PATCH 1: one-time ISO timestamp backfill
    if st.button("Backfill ISO timestamps (created_at / updated_at)"):
        res = backfill_iso_timestamps(engine)
        st.success(f"Backfill complete — created_at: {res['created_at']}, updated_at: {res['updated_at']}")

    _csv_restore_append(engine)
    _quick_probes(engine)

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
# ==== BEGIN DROP-IN: page_admin_taxonomy (two-column layout; no page-level subheader) ====
def page_admin_taxonomy(engine: Engine):
    # Fetch once per render
    categories = fetch_categories(engine)
    services = fetch_services(engine)

    col_cat, col_svc = st.columns(2)

    # ------------- LEFT: Category Admin -------------
    with col_cat:
        st.markdown("#### Category Admin")

        # Add
        st.markdown("**Add Category**")
        with st.form("cat_add"):
            new_cat = st.text_input("New category name", key="new_cat")
            add_cat = st.form_submit_button("Add Category", type="primary")
            if add_cat:
                name = (new_cat or "").strip()
                if not name:
                    st.error("Category name required.")
                elif name.lower() in {c.lower() for c in categories}:
                    st.warning("Category already exists.")
                else:
                    with engine.begin() as conn:
                        conn.execute(sql_text("INSERT INTO categories(name) VALUES(:n)"), {"n": name})
                    st.success(f"Added category '{name}'.")
                    st.rerun()

        st.markdown("---")

        # Rename (cascade to vendors)
        st.markdown("**Rename Category (cascades to vendors)**")
        with st.form("cat_rename"):
            old = st.selectbox("Current category", options=[""] + categories, index=0, key="old_cat")
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

        # Delete (only if unused)
        st.markdown("**Delete Category (only if unused)**")
        with st.form("cat_delete"):
            target = st.selectbox("Category to delete", options=[""] + categories, index=0, key="del_cat")
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

    # ------------- RIGHT: Service Admin -------------
    with col_svc:
        st.markdown("#### Service Admin")

        # Add
        st.markdown("**Add Service**")
        with st.form("svc_add"):
            new_svc = st.text_input("New service name", key="new_svc")
            add_svc = st.form_submit_button("Add Service", type="primary")
            if add_svc:
                name = (new_svc or "").strip()
                if not name:
                    st.error("Service name required.")
                elif name.lower() in {s.lower() for s in services}:
                    st.warning("Service already exists.")
                else:
                    with engine.begin() as conn:
                        conn.execute(sql_text("INSERT INTO services(name) VALUES(:n)"), {"n": name})
                    st.success(f"Added service '{name}'.")
                    st.rerun()

        st.markdown("---")

        # Rename (cascade to vendors)
        st.markdown("**Rename Service (cascades to vendors)**")
        with st.form("svc_rename"):
            old = st.selectbox("Current service", options=[""] + services, index=0, key="old_svc")
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

        # Delete (only if unused)
        st.markdown("**Delete Service (only if unused)**")
        with st.form("svc_delete"):
            target = st.selectbox("Service to delete", options=[""] + services, index=0, key="del_svc")
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
# ==== END DROP-IN: page_admin_taxonomy ====


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
