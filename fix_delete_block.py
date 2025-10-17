from pathlib import Path
import re
p = Path("app_admin.py")
s = p.read_text(encoding="utf-8")

# Robust markers (allow spaces and both ' or " quotes, and varying hyphens)
start_re = re.compile(r'^[ \t]*st\.subheader\(\s*["\']Delete Provider["\']\s*\)', re.M)
end_re   = re.compile(r'^[ \t]*#\s*-+\s*Category Admin', re.M)

m_start = start_re.search(s)
if not m_start:
    # Try a looser fallback: any subheader with "Delete" and "Provider" nearby
    start_re = re.compile(r'^[ \t]*st\.subheader\(\s*["\'].*Delete.*Provider.*["\']\s*\)', re.M | re.I)
    m_start = start_re.search(s)

if not m_start:
    print("ERROR: Start marker not found.")
    raise SystemExit(1)

m_end = end_re.search(s, m_start.end())
if not m_end:
    print("ERROR: End marker not found.")
    raise SystemExit(1)

backup = p.with_suffix(".before_delete_patch.py")
backup.write_text(s, encoding="utf-8")

new_block = '''st.markdown("---")
st.subheader("Delete Provider")

sel_label_del = st.selectbox(
    "Select provider to delete (type to search)",
    options=["— Select —"] + [ _fmt_vendor(i) for i in ids ],
    key="delete_provider_label",
)
if sel_label_del != "— Select —":
    rev = { _fmt_vendor(i): i for i in ids }
    st.session_state["delete_vendor_id"] = int(rev.get(sel_label_del))
else:
    st.session_state["delete_vendor_id"] = None

del_form_key = f"delete_vendor_form_{st.session_state['delete_form_version']}"
with st.form(del_form_key, clear_on_submit=False):
    deleted = st.form_submit_button("Delete Provider")

if deleted:
    del_nonce = _nonce("delete")
    if st.session_state.get("delete_last_done") == del_nonce:
        st.info("Delete already processed.")
        st.stop()

    vid = st.session_state.get("delete_vendor_id")
    if vid is None:
        st.error("Select a provider first.")
    else:
        try:
            row = df_all.loc[df_all["id"] == int(vid)]
            prev_updated = (row.iloc[0]["updated_at"] if not row.empty else "") or ""
            res = _exec_with_retry(
                ENGINE,
                """
                DELETE FROM vendors
                 WHERE id=:id AND (updated_at=:prev_updated OR :prev_updated='')
                """,
                {"id": int(vid), "prev_updated": prev_updated},
            )
            rowcount = int(getattr(res, "rowcount", 0) or 0)
            if rowcount == 0:
                st.warning("No delete performed (stale selection). Refresh and try again.")
            else:
                st.session_state["delete_last_done"] = del_nonce
                st.success("Provider deleted.")
                _queue_delete_form_reset()
                _nonce_rotate("delete")
                list_names.clear()
                st.rerun()
        except Exception as e:
            st.error(f"Delete failed: {e}")

# ---------- Category Admin
'''

s2 = s[:m_start.start()] + new_block + s[m_end.start():]
p.write_text(s2, encoding="utf-8")
print("OK: Delete Provider block replaced.")
