-- ==== BEGIN: providers seed.sql (replace file) ====
-- Tables
CREATE TABLE IF NOT EXISTS categories (
  id   INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS services (
  id   INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS vendors (
  id              INTEGER PRIMARY KEY,
  category        TEXT NOT NULL DEFAULT '',
  service         TEXT NOT NULL DEFAULT '',
  business_name   TEXT NOT NULL DEFAULT '',
  contact_name    TEXT NOT NULL DEFAULT '',
  phone           TEXT NOT NULL DEFAULT '',
  address         TEXT NOT NULL DEFAULT '',
  website         TEXT NOT NULL DEFAULT '',
  notes           TEXT NOT NULL DEFAULT '',
  keywords        TEXT NOT NULL DEFAULT '',
  created_at      TEXT NOT NULL DEFAULT '',
  updated_at      TEXT NOT NULL DEFAULT '',
  updated_by      TEXT NOT NULL DEFAULT '',
  computed_keywords TEXT NOT NULL DEFAULT ''
);

-- Indexes (performance mirrors runtime)
CREATE INDEX IF NOT EXISTS idx_vendors_cat        ON vendors(category);
CREATE INDEX IF NOT EXISTS idx_vendors_svc        ON vendors(service);
CREATE INDEX IF NOT EXISTS idx_vendors_bus        ON vendors(business_name);
CREATE INDEX IF NOT EXISTS idx_vendors_kw         ON vendors(keywords);
CREATE INDEX IF NOT EXISTS idx_vendors_phone      ON vendors(phone);
CREATE INDEX IF NOT EXISTS idx_vendors_cat_lower  ON vendors(LOWER(category));
CREATE INDEX IF NOT EXISTS idx_vendors_svc_lower  ON vendors(LOWER(service));
CREATE INDEX IF NOT EXISTS idx_vendors_bus_lower  ON vendors(LOWER(business_name));
CREATE INDEX IF NOT EXISTS idx_vendors_ckw        ON vendors(computed_keywords);

-- Optional one-time backfills (safe, idempotent)
UPDATE vendors
   SET created_at = COALESCE(NULLIF(created_at,''), updated_at, strftime('%Y-%m-%dT%H:%M:%SZ','now')),
       updated_at = COALESCE(NULLIF(updated_at,''), strftime('%Y-%m-%dT%H:%M:%SZ','now'))
 WHERE (created_at = '' OR updated_at = '');

UPDATE vendors
   SET category=COALESCE(category,''), service=COALESCE(service,''), business_name=COALESCE(business_name,''),
       contact_name=COALESCE(contact_name,''), phone=COALESCE(phone,''), address=COALESCE(address,''),
       website=COALESCE(website,''), notes=COALESCE(notes,''), keywords=COALESCE(keywords,''),
       updated_by=COALESCE(updated_by,''), computed_keywords=COALESCE(computed_keywords,'');
-- ==== END: providers seed.sql (replace file) ====

