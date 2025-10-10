## How I Make Changes Safely

This repo uses a simple, low-drama workflow that keeps `main` clean and the Streamlit apps stable.

### One-time repo settings (already done)
- **Branch protection on `main`**: PR required, approvals = 0, conversation resolution = on, linear history = on, block force pushes = on.
- **Merge methods**: Allow **Squash** (recommended). (Optional: allow Rebase.) **Do not** allow merge commits.
- Streamlit Cloud apps point to **`main`**, and secrets live **only** in Cloud (not in the repo).

---

### Everyday change flow (UI only)

1. **Start from `main`**
   - Code tab → branch dropdown → select `main` (refresh so it’s up-to-date).

2. **Create a *new* topic branch**
   - Branch dropdown → type a name → **Create branch**.  
   - Naming:
     - `feat/<what-you-add>` (feature)
     - `fix/<what-you-fix>` (bug)
     - `chore/<what-you-tidy>` (format/infra)
   - Examples: `fix/phone-normalize`, `feat/csv-import`, `chore/formatting-2025-10-05`.

3. **Edit on that branch**
   - Open the file → **✏️ Edit** → make the change.
   - Commit message: short and specific, e.g., `fix: Browse tab indent`.
   - Ensure you commit **to your branch**, not `main`.

4. **Open a Pull Request**
   - Click **Compare & pull request**.
   - Base: `main` ← Compare: *your branch*.
   - Review the diff (only the lines you intended should change).
   - Create PR.

5. **Merge cleanly**
   - Use **Squash and merge**.
   - **Delete the branch** when prompted (keeps things tidy).

6. **Verify deployment**
   - Streamlit apps auto-redeploy from `main`. Open Admin & Read-only → quick smoke test.

7. **(Optional) Tag a release**
   - For milestones: Releases → “Draft a new release” → Tag like `vYYYY-MM-DD`.

---

### Data backup policy (vendors, categories, services)

> Code is versioned by Git; **data** is not. Always back up data before risky changes.

- **Quick CSV backup** (from Admin app):  
  Maintenance → **Export all vendors (CSV)** → save as `backups/providers-YYYY-MM-DD.csv`.  
  ⚠️ If the repo is **public**, do not commit CSVs with phone/address; store them privately.

- **Full SQL dump** (optional, terminal):  
  `turso db shell vendors-prod --exec ".dump" > backup-YYYYMMDD.sql` *(read-only)*

---

### Rollback recipes

- **Revert a PR**: open the merged PR → **Revert** → merge the revert PR.
- **Restore a single file**: README → file **History** → open last good commit → copy → new small branch → paste → PR → merge.
- **Compare to a release**: Releases → pick tag (e.g., `vYYYY-MM-DD`) → compare to current to see differences.

---

### Rules that prevent mess

- **Do**
  - One topic per branch/PR
  - Keep branches short-lived
  - Clear commit/PR titles
  - Export data before risky changes

- **Don’t**
  - Edit on `main`
  - Reuse old branches for new work
  - Commit secrets (keep them in Streamlit Cloud)
  - Enable required CI checks unless CI exists (it will block merges)

---

### FAQ

- **Why Squash merges?** One tidy commit per PR, easy to revert, linear history.
- **Why approvals = 0?** Solo maintainer—GitHub won’t count your own approval.
- **What about status checks?** Leave off unless you’ve set up CI; otherwise PRs will be blocked.
- **Where do secrets live?** Only in Streamlit Cloud → App Secrets (never in the repo).
## Quick Smoke Test (Pre-Deploy)
Run a 1-minute database check before pushing to Streamlit Cloud:
```bash
python scripts/smoke.py
