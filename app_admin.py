# app_admin.py
# -*- coding: utf-8 -*-
from __future__ import annotations

# ---- Page config MUST be first streamlit call ----
import streamlit as st
st.set_page_config(page_title="HCR Providers — Admin", page_icon="🛠️", layout="wide")

import os, re, time, sys, html, json, textwrap
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# Ensure the libsql dialect is registered (sqlite+libsql)
try:
    import sqlalchemy_libsql  # noqa: F401
except Exception:
    pass

# =============================
# Configuration / Constants
# =============================

APP_VER = "admin-2025-10-16.1"
CKW_VERSION = "ckw-2025-10-16a"  # bump when generator changes
MAX_ROWS = 1000

# =============================
# Secrets helpers
# =============================

def _get_secret(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

def _as_bool(v, default=False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in ("1","true","t","yes","y","on")

SHOW_STATUS = _as_bool(_get_secret("ADMIN_SHOW_STATUS", True), True)
SHOW_DIAGS  = _as_bool(_get_secret("ADMIN_SHOW_DIAGS", False), False)
DB_STRATEGY = str(_get_secret("DB_STRATEGY", "embedded_replica")).strip().lower()

TURSO_URL   = _get_secret("TURSO_DATABASE_URL", "")
TURSO_TOKEN = _get_secret("TURSO_AUTH_TOKEN", "")
EMBEDDED_PATH = _get_secret("EMBEDDED_DB_PATH", "vendors-embedded.db")

# =============================
# Engine / Retry
# =============================

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

@st.cache_resource(show_spinner=False)
def build_engine() -> Engine:
    """
    Build a stable engine suitable for libsql embedded replica.
    """
    if DB_STRATEGY == "embedded_replica":
        if not os.path.isabs(EMBEDDED_PATH):
            embedded = os.path.join(os.getcwd(), EMBEDDED_PATH)
        else:
            embedded = EMBEDDED_PATH
        url = f"sqlite+libsql:///{embedded}"
        os.makedirs(os.path.dirname(embedded) or ".", exist_ok=True)
    elif DB_STRATEGY == "turso_only":
        url = TURSO_URL
    else:
        url = f"sqlite+libsql:///{EMBEDDED_PATH}"

    if SHOW_STATUS:
        st.sidebar.info(f"Engine URL scheme ok: {bool(url)} | Strategy: {DB_STRATEGY}")

    eng = create_engine(url, pool_pre_ping=True, pool_recycle=300)
    return eng

def _exec_with_retry(engine: Engine, sql: str, params: Optional[dict]=None, tries: int=3, delay: float=0.25):
    last = None
    for i in range(tries):
        try:
            with engine.begin() as cx:
                return cx.execute(sql_text(sql), params or {})
        except Exception as e:
            last = e
            time.sleep(delay * (2**i))
    raise last

# =============================
# Schema / Ensure
# =============================

DDL_META = """
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);
"""

DDL_VENDORS = """
CREATE TABLE IF NOT EXISTS vendors (
  id               INTEGER PRIMARY KEY,
  category         TEXT,
  service          TEXT,
  business_name    TEXT,
  contact_name     TEXT,
  phone            TEXT,          -- digits-only storage
  address          TEXT,
  website          TEXT,
  notes            TEXT,
  keywords         TEXT,          -- user-entered free-text
  computed_keywords TEXT,         -- auto-generated
  ckw_version      TEXT,
  deleted_at       TEXT,          -- soft delete ISO
  created_at       TEXT,
  updated_at       TEXT,
  updated_by       TEXT
);
"""

DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_vendors_bn      ON vendors(business_name);",
    "CREATE INDEX IF NOT EXISTS idx_vendors_cat     ON vendors(category);",
    "CREATE INDEX IF NOT EXISTS idx_vendors_svc     ON vendors(service);",
    "CREATE INDEX IF NOT EXISTS idx_vendors_ckw     ON vendors(computed_keywords);",
    "CREATE INDEX IF NOT EXISTS idx_vendors_deleted ON vendors(deleted_at);",
]

DDL_CKW_SEEDS = """
CREATE TABLE IF NOT EXISTS ckw_seeds (
  category TEXT NOT NULL,
  service  TEXT NOT NULL,
  seed     TEXT NOT NULL,
  PRIMARY KEY (category, service)
);
"""

def ensure_schema(engine: Engine):
    _exec_with_retry(engine, DDL_META)
    _exec_with_retry(engine, DDL_VENDORS)
    for ddl in DDL_INDEXES:
        _exec_with_retry(engine, ddl)
    _exec_with_retry(engine, DDL_CKW_SEEDS)
    # meta keys
    _exec_with_retry(engine, "INSERT OR IGNORE INTO meta(key,value) VALUES('data_version','0');")
    _exec_with_retry(engine, "INSERT OR IGNORE INTO meta(key,value) VALUES('last_maintenance','');")

# =============================
# Helpers: normalization / CKW
# =============================

def _digits_only(s: Optional[str]) -> str:
    return "".join(ch for ch in (s or "") if ch.isdigit())[:15]

def _sanitize_url(s: Optional[str]) -> str:
    u = (s or "").strip()
    if not u:
        return ""
    if not re.match(r"^https?://", u, re.IGNORECASE):
        u = "https://" + u
    return u

def _kw_split(s: str) -> List[str]:
    # split by comma, semicolon, whitespace; lower+dedupe
    tokens = re.split(r"[,\s;]+", (s or "").strip().lower())
    return sorted({t for t in tokens if t})

def _kw_join(tokens: List[str]) -> str:
    return ", ".join(sorted({t.strip().lower() for t in tokens if t.strip()}))

def _ckw_seed_map(engine: Engine) -> Dict[Tuple[str,str], str]:
    rows = _exec_with_retry(engine, "SELECT category, service, seed FROM ckw_seeds").fetchall()
    return {(r[0] or "", r[1] or ""): (r[2] or "") for r in rows}

def _gen_ckw(category: str, service: str, business_name: str, seeds: Dict[Tuple[str,str], str], manual_keywords: str) -> str:
    """
    Deterministic generator. Merge:
    - curated seeds by (category, service)
    - business_name tokens (selective)
    - user free-text keywords
    """
    cat = (category or "").strip()
    svc = (service or "").strip()
    bn  = (business_name or "").strip()

    seed = seeds.get((cat, svc), "")
    parts = []
    parts.extend(_kw_split(seed))
    parts.extend([w for w in _kw_split(bn) if len(w) >= 3])
    parts.extend(_kw_split(manual_keywords))
    return _kw_join(parts)[:400]  # keep it tidy

def _updated_by() -> str:
    return _get_secret("ADMIN_UPDATED_BY", "admin")

# =============================
# Cached loads (keyed by data_version)
# =============================

def _get_data_version(engine: Engine) -> str:
    try:
        with engine.connect() as cx:
            v = cx.execute(sql_text("SELECT value FROM meta WHERE key='data_version'")).scalar()
            return str(v or "0")
    except Exception:
        return "0"
@st.cache_data(show_spinner=False)
def load_active_df(_engine: Engine, version: str) -> pd.DataFrame:
    q = """
    SELECT id, category, service, business_name, contact_name, phone, address,
           website, notes, keywords, computed_keywords, ckw_version,
           created_at, updated_at, updated_by
      FROM vendors
     WHERE deleted_at IS NULL
     ORDER BY business_name COLLATE NOCASE
    """
    with _engine.connect() as cx:
        df = pd.read_sql(sql_text(q), cx)
    return df


@st.cache_data(show_spinner=False)
def load_deleted_df(engine: Engine, version: str) -> pd.DataFrame:
    q = """
    SELECT id, category, service, business_name, contact_name, phone, address,
           website, notes, keywords, computed_keywords, ckw_version,
           created_at, updated_at, updated_by, deleted_at
      FROM vendors
     WHERE deleted_at IS NOT NULL
     ORDER BY business_name COLLATE NOCASE
    """
    with engine.connect() as cx:
        df = pd.read_sql(sql_text(q), cx)
    return df

def _bump_data_version(engine: Engine):
    _exec_with_retry(engine, "UPDATE meta SET value=:v WHERE key='data_version'", {"v": _now_utc_iso()})
    # Clear only data caches
    load_active_df.clear()
    load_deleted_df.clear()

# =============================
# Auto-Maintenance
# =============================

def run_auto_maintenance(engine: Engine, seeds: Dict[Tuple[str,str], str]) -> dict:
    """
    Recompute CKW where missing/outdated. Normalize digits in phone.
    One pass over ≤1k rows; returns stats.
    """
    stats = {"rows":0, "ckw_updates":0, "phone_fixes":0}
    q = """
    SELECT id, category, service, business_name, keywords, computed_keywords, ckw_version, phone
      FROM vendors
     WHERE deleted_at IS NULL
    """
    updates = []
    with engine.connect() as cx:
        rows = cx.execute(sql_text(q)).fetchall()
    stats["rows"] = len(rows)
    for r in rows:
        vid, cat, svc, bn, kw, ckw, ver, ph = r
        # phone normalization (idempotent)
        ph_norm = _digits_only(ph)
        need_phone = (ph_norm != (ph or ""))

        # CKW recompute if missing or version mismatch
        gen = _gen_ckw(cat or "", svc or "", bn or "", seeds, kw or "")
        need_ckw = (not ckw) or (ckw.strip() == "") or (ver != CKW_VERSION)

        if need_ckw or need_phone:
            updates.append((vid, gen if need_ckw else (ckw or ""), CKW_VERSION if need_ckw else (ver or ""), ph_norm if need_phone else (ph or "")))
            if need_ckw: stats["ckw_updates"] += 1
            if need_phone: stats["phone_fixes"] += 1

    if updates:
        with engine.begin() as cx:
            for (vid, ckw_new, ver_new, ph_new) in updates:
                cx.execute(sql_text("""
                    UPDATE vendors
                       SET computed_keywords=:ckw,
                           ckw_version=:ver,
                           phone=:ph
                     WHERE id=:id
                """), {"id": vid, "ckw": ckw_new, "ver": ver_new, "ph": ph_new})
        _exec_with_retry(engine, "UPDATE meta SET value=:ts WHERE key='last_maintenance'", {"ts": _now_utc_iso()})
        _bump_data_version(engine)
    else:
        _exec_with_retry(engine, "UPDATE meta SET value=:ts WHERE key='last_maintenance'", {"ts": _now_utc_iso()})
    return stats

# =============================
# UI Helpers
# =============================

def _fmt_phone_display(s: str) -> str:
    d = _digits_only(s)
    if len(d) == 10:
        return f"({d[:3]}) {d[3:6]}-{d[6:]}"
    return d

def _info_banner(engine: Engine):
    if not SHOW_STATUS: 
        return
    try:
        with engine.connect() as cx:
            total = cx.execute(sql_text("SELECT COUNT(*) FROM vendors")).scalar() or 0
            deleted = cx.execute(sql_text("SELECT COUNT(*) FROM vendors WHERE deleted_at IS NOT NULL")).scalar() or 0
            lv = cx.execute(sql_text("SELECT value FROM meta WHERE key='data_version'")).scalar() or "0"
            lm = cx.execute(sql_text("SELECT value FROM meta WHERE key='last_maintenance'")).scalar() or ""
        st.sidebar.success(f"Rows: {total} (active {total - deleted}, deleted {deleted})\n\ndata_version: {lv}\nlast_maint: {lm}\nver: {APP_VER}")
    except Exception as e:
        st.sidebar.warning(f"Status failed: {e}")

def _render_table(df: pd.DataFrame):
    if df.empty:
        st.info("No matching providers.")
        return
    # Tight HTML table for speed; format only visible (≤1000)
    disp = df.copy()
    if "phone" in disp.columns:
        disp["phone"] = disp["phone"].map(_fmt_phone_display)
    cols = [c for c in disp.columns if c not in ("computed_keywords","ckw_version")]
    st.dataframe(disp[cols], use_container_width=True, hide_index=True)

# =============================
# Main
# =============================

def main():
    st.title("Providers — Admin")
    engine = build_engine()
    ensure_schema(engine)
    _info_banner(engine)

    # Seeds & Auto-Maintenance on startup
    seeds = _ckw_seed_map(engine)
    maint_stats = run_auto_maintenance(engine, seeds)  # quick; ≤1k rows

    tabs = st.tabs(["Browse", "Add / Edit / Delete", "Maintenance"])
    # ---- Browse ----
    with tabs[0]:
        version = _get_data_version(engine)
        df = load_active_df(engine, version)

        q = st.text_input("Search", key="q_admin", placeholder="name, category, service, notes, keywords…")
        qn = (q or "").strip().lower()
        if qn:
            mask = (
                df["business_name"].str.lower().str.contains(qn, na=False) |
                df["category"].str.lower().str.contains(qn, na=False) |
                df["service"].str.lower().str.contains(qn, na=False) |
                df["notes"].str.lower().str.contains(qn, na=False) |
                df["keywords"].str.lower().str.contains(qn, na=False) |
                df["computed_keywords"].str.lower().str.contains(qn, na=False)
            )
            df_view = df[mask]
        else:
            df_view = df

        st.caption(f"{len(df_view)}/{len(df)} rows")
        _render_table(df_view)

        with st.expander("Show deleted"):
            dfd = load_deleted_df(engine, version)
            _render_table(dfd)
            if not dfd.empty:
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Restore selected by ID", type="secondary"):
                        st.info("Use the Edit tab to restore by ID (safer).")
                with c2:
                    pass  # keep simple here

    # ---- Add / Edit / Delete ----
    with tabs[1]:
        version = _get_data_version(engine)
        df = load_active_df(engine, version)

        st.subheader("Select a provider to edit, or add a new one")
        col_sel, col_btn = st.columns([3,1])
        with col_sel:
            options = ["(Add New)"] + [f"{int(r.id)} — {r.business_name}" for r in df.itertuples()]
            choice = st.selectbox("Record", options, key="edit_selector")
        with col_btn:
            st.write("")
            st.write("")
            if st.button("Add New"):
                st.session_state["edit_selector"] = "(Add New)"
                st.rerun()

        is_new = (choice == "(Add New)")
        current = {}
        if not is_new and choice:
            vid = int(choice.split(" — ",1)[0])
            rows = df[df["id"]==vid]
            if not rows.empty:
                current = rows.iloc[0].to_dict()

        with st.form("edit_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                business_name = st.text_input("Provider *", value=(current.get("business_name","") if current else ""))
                category      = st.text_input("Category", value=(current.get("category","") if current else ""))
                service       = st.text_input("Service", value=(current.get("service","") if current else ""))
                contact_name  = st.text_input("Contact", value=(current.get("contact_name","") if current else ""))
                phone_in      = st.text_input("Phone (digits only stored)", value=(current.get("phone","") if current else ""))

            with c2:
                address = st.text_area("Address", height=80, value=(current.get("address","") if current else ""))
                website = st.text_input("Website (https://…)", value=(current.get("website","") if current else ""))
                notes   = st.text_area("Notes", height=100, value=(current.get("notes","") if current else ""))
                keywords= st.text_input("Keywords (comma separated)", value=(current.get("keywords","") if current else ""))

                # ---- CKW Preview (auto; no editor) ----
                seeds = _ckw_seed_map(engine)  # small table; fine to read here
                suggested_now = _gen_ckw(category, service, business_name, seeds, keywords)
                st.caption("Computed Keywords (auto-generated on Save)")
                st.code(suggested_now or "(none)", language="text")

            c3, c4, c5 = st.columns([1,1,1])
            with c3:
                saved = st.form_submit_button("Save", type="primary")
            with c4:
                delete_click = st.form_submit_button("Delete (soft)")
            with c5:
                purge_click = st.form_submit_button("Purge forever (type name first)")

        if saved:
            bn = (business_name or "").strip()
            if not bn:
                st.error("Provider name is required.")
            else:
                now = _now_utc_iso()
                phone_norm = _digits_only(phone_in)
                website_s  = _sanitize_url(website)
                if is_new:
                    _exec_with_retry(engine, """
                        INSERT INTO vendors(category, service, business_name, contact_name, phone, address,
                                            website, notes, keywords, computed_keywords, ckw_version,
                                            created_at, updated_at, updated_by)
                        VALUES(:cat,:svc,:bn,:cn,:ph,:addr,:web,:notes,:kw,:ckw,:ver,:now,:now,:user)
                    """, {
                        "cat": category, "svc": service, "bn": bn, "cn": contact_name, "ph": phone_norm,
                        "addr": address, "web": website_s, "notes": notes, "kw": keywords,
                        "ckw": suggested_now, "ver": CKW_VERSION, "now": now, "user": _updated_by()
                    })
                else:
                    _exec_with_retry(engine, """
                        UPDATE vendors
                           SET category=:cat, service=:svc, business_name=:bn, contact_name=:cn, phone=:ph,
                               address=:addr, website=:web, notes=:notes, keywords=:kw,
                               computed_keywords=:ckw, ckw_version=:ver,
                               updated_at=:now, updated_by=:user
                         WHERE id=:id
                    """, {
                        "id": int(vid),
                        "cat": category, "svc": service, "bn": bn, "cn": contact_name, "ph": phone_norm,
                        "addr": address, "web": website_s, "notes": notes, "kw": keywords,
                        "ckw": suggested_now, "ver": CKW_VERSION, "now": now, "user": _updated_by()
                    })
                _bump_data_version(engine)
                st.success("Saved.")
                st.rerun()

        if delete_click and not is_new:
            now = _now_utc_iso()
            _exec_with_retry(engine, "UPDATE vendors SET deleted_at=:ts, updated_at=:ts, updated_by=:u WHERE id=:id",
                             {"ts": now, "u": _updated_by(), "id": int(vid)})
            _bump_data_version(engine)
            st.success("Record soft-deleted. Use 'Show deleted' to restore or Maintenance to purge.")
            st.rerun()

        if purge_click and not is_new:
            # simple safety: require exact name typed in a text_input (modal-less)
            check = st.text_input("Type the Provider name EXACTLY to confirm purge:", key=f"purge_check_{vid}")
            if check == (current.get("business_name") or ""):
                _exec_with_retry(engine, "DELETE FROM vendors WHERE id=:id", {"id": int(vid)})
                _bump_data_version(engine)
                st.success("Purged.")
                st.rerun()
            else:
                st.warning("Name did not match; purge cancelled.")

    # ---- Maintenance ----
    with tabs[2]:
        st.subheader("Maintenance")
        st.caption("Auto-Maintenance already ran on startup. Use this if you changed seeds or after CSV append.")
        if st.button("Run Maintenance Now", type="secondary"):
            seeds = _ckw_seed_map(engine)
            stats = run_auto_maintenance(engine, seeds)
            st.success(f"Done. {stats}")
        if SHOW_DIAGS:
            with st.expander("Diagnostics"):
                st.write({"APP_VER": APP_VER, "CKW_VERSION": CKW_VERSION, "DB_STRATEGY": DB_STRATEGY})
                st.code(json.dumps(maint_stats, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Admin app crashed. See logs for details.")
        st.exception(e)
