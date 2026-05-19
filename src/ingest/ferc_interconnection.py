#!/usr/bin/env python3
"""
ferc_interconnection.py — ISO/RTO interconnection queue ingestion.

Pulls active interconnection queue files from all 7 major US ISOs,
filters for datacenter-related load entries, writes to warehouse table
`ferc_interconnection_datacenter`.

ISOs covered: PJM, MISO, CAISO, ERCOT, NYISO, ISO-NE, SPP

Methodology:
  Keyword search across project name / applicant / description columns for:
  data center, datacenter, hyperscale, cloud, colocation, colo, campus, ai

If an ISO's queue cannot be programmatically downloaded, the failure is
documented and that ISO is skipped — no synthetic data is substituted.

Usage:
  python src/ingest/ferc_interconnection.py
"""
import io, json, requests, pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from db import get_conn

ROOT      = Path(__file__).parent.parent.parent
CACHE_DIR = ROOT / "data" / "raw" / "ferc_queues"

DC_KEYWORDS = [
    "data center", "datacenter", "data-center",
    "hyperscale", "hyper scale",
    "cloud", "colocation", "colo",
    "campus",
    "artificial intelligence", " ai ",
    "computing", "server farm",
]

# ---------------------------------------------------------------------------
# ISO source registry
# Each entry: url (str), fmt ('xlsx'|'csv'|'zip'), note (access status)
# ---------------------------------------------------------------------------
ISO_SOURCES: dict[str, dict] = {
    "PJM": {
        # PJM now requires browser session — go to pjm.com/planning/interconnection-projects/interconnection-queue
        # click "Download Queue" and save as data/raw/ferc_queues/pjm_queue.xlsx
        "url": "https://www.pjm.com/-/media/planning/interqueue/queue.ashx",
        "fmt": "xlsx",
        "name_cols": ["Project Name", "Applicant", "Fuel"],
        "mw_col": "MW In Service",
        "state_col": "State",
        "date_col": "Queue Date",
        "status_col": "Status",
        "note": "MANUAL: pjm.com → Planning → Interconnection Queue → Download Queue",
    },
    "MISO": {
        # Go to misoenergy.org/planning/resource-interconnection/interconnection-queue
        # Active Projects tab → Export to Excel → save as miso_queue.xlsx
        "url": "https://www.misoenergy.org/api/download?type=1&mime=application/xlsx",
        "fmt": "xlsx",
        "name_cols": ["Project Name", "Fuel Type", "Sponsor"],
        "mw_col": "Capacity (MW)",
        "state_col": "State",
        "date_col": "Queue Date",
        "status_col": "Queue Status",
        "note": "MANUAL: misoenergy.org → Planning → Interconnection Queue → Active Projects → Export",
    },
    "CAISO": {
        # Go to caiso.com → Markets & Operations → Generator Interconnection
        # Download "Interconnection Queue" Excel → save as caiso_queue.xlsx
        "url": "https://www.caiso.com/Documents/InterconnectionQueue.xlsx",
        "fmt": "xlsx",
        "name_cols": ["Project Name", "Fuel Type", "Applicant"],
        "mw_col": "MW",
        "state_col": "County",
        "date_col": "Application Received",
        "status_col": "Current Step",
        "note": "MANUAL: caiso.com → Markets & Operations → Generator Interconnection → Queue download",
    },
    "ERCOT": {
        # ERCOT GIS Report: ercot.com → Grid Info → Resource Integration → GIS Report
        # Download the Excel file → save as ercot_queue.xlsx
        "url": "https://mis.ercot.com/misapp/GetReports.do?reportTypeId=15933",
        "fmt": "xlsx",
        "name_cols": ["Project Name", "Technology", "Applicant"],
        "mw_col": "MW",
        "state_col": "County",
        "date_col": "Application Date",
        "status_col": "Status",
        "note": "MANUAL: ercot.com → Grid Info → Resource Integration → GIS Report → Download Excel",
    },
    "NYISO": {
        # NYISO: nyiso.com → Interconnections → Interconnection Queue
        # Download Excel → save as nyiso_queue.xlsx
        "url": "https://www.nyiso.com/documents/20142/2226394/NYISO-Interconnection-Queue.xlsx/",
        "fmt": "xlsx",
        "name_cols": ["Project Name", "Fuel Type", "Developer"],
        "mw_col": "MW In Service",
        "state_col": "County",
        "date_col": "Queue Date",
        "status_col": "Status",
        "note": "MANUAL: nyiso.com → Interconnections → Queue → Download Excel",
    },
    "ISO-NE": {
        # ISO-NE: iso-ne.com → System Planning → Interconnection → Queue
        # Download the current queue Excel → save as iso_ne_queue.xlsx
        "url": "https://www.iso-ne.com/static-assets/documents/grid/gen-q/gen_q_queue.xlsx",
        "fmt": "xlsx",
        "name_cols": ["Project Name", "Fuel Type", "Applicant"],
        "mw_col": "Capacity (MW)",
        "state_col": "State",
        "date_col": "Queue Date",
        "status_col": "Status",
        "note": "MANUAL: iso-ne.com → System Planning → Interconnection Service → Request Queue → Download",
    },
    "SPP": {
        # SPP: spp.org → Engineering → Generator Interconnection → Interconnection Queue
        # Download Excel → save as spp_queue.xlsx
        "url": (
            "https://www.spp.org/documents/0/0/GRID%20INFORMATION/"
            "NETWORK%20TRANSMISSION/GENERATION%20INTERCONNECTION%20QUEUE/"
            "SPP_GI_QUEUE.xlsx"
        ),
        "fmt": "xlsx",
        "name_cols": ["Project Name", "Fuel Type", "Developer/Applicant"],
        "mw_col": "MW",
        "state_col": "State",
        "date_col": "Queue Date",
        "status_col": "Status",
        "note": "MANUAL: spp.org → Engineering → Generator Interconnection → Queue → Download",
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/octet-stream,*/*",
}


def _download(iso: str, url: str) -> bytes | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{iso.lower().replace('-','_')}_queue.xlsx"
    if cache.exists():
        print(f"   Using cached {cache.name}")
        return cache.read_bytes()
    try:
        r = requests.get(url, timeout=60, headers=HEADERS, allow_redirects=True)
        r.raise_for_status()
        if len(r.content) < 5000:
            raise ValueError(f"Response too small ({len(r.content)} bytes) — likely a login page")
        cache.write_bytes(r.content)
        print(f"   Downloaded {len(r.content) / 1e6:.1f} MB → {cache.name}")
        return r.content
    except Exception as e:
        print(f"   ⚠️  {iso} download failed: {e}")
        print(f"      Manual download: {url}")
        return None


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # fuzzy: any column containing the candidate substring
    for cand in candidates:
        for col in df.columns:
            if cand.lower() in col.lower():
                return col
    return None


def _filter_datacenters(df: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for col in text_cols:
        if col in df.columns:
            col_str = df[col].fillna("").astype(str).str.lower()
            for kw in DC_KEYWORDS:
                mask |= col_str.str.contains(kw, regex=False)
    return df[mask]


def parse_queue(iso: str, data: bytes, cfg: dict) -> pd.DataFrame | None:
    try:
        df = pd.read_excel(io.BytesIO(data), engine="openpyxl")
    except Exception:
        try:
            df = pd.read_excel(io.BytesIO(data), engine="xlrd")
        except Exception as e:
            print(f"   ⚠️  Could not parse {iso} queue: {e}")
            return None

    if df.empty:
        print(f"   ⚠️  {iso} queue file is empty")
        return None

    # Resolve configured column names against actual headers
    name_cols   = [c for c in cfg["name_cols"] if _find_col(df, [c])]
    name_cols   = [_find_col(df, [c]) for c in cfg["name_cols"] if _find_col(df, [c])]
    mw_col      = _find_col(df, [cfg["mw_col"]])
    state_col   = _find_col(df, [cfg["state_col"]])
    date_col    = _find_col(df, [cfg["date_col"]])
    status_col  = _find_col(df, [cfg["status_col"]])

    print(f"   {iso}: {len(df)} rows, {len(df.columns)} cols")

    # Filter for datacenter keywords across all text columns
    dc = _filter_datacenters(df, name_cols + [c for c in [state_col] if c])
    print(f"   {iso}: {len(dc)} datacenter-matching rows")

    if dc.empty:
        return None

    # Normalise to a standard output schema
    out = pd.DataFrame()
    out["iso"]          = iso
    out["project_name"] = dc[name_cols[0]].astype(str) if name_cols else "unknown"
    out["capacity_mw"]  = pd.to_numeric(dc[mw_col], errors="coerce") if mw_col else None
    out["state"]        = dc[state_col].astype(str) if state_col else None
    out["queue_date"]   = pd.to_datetime(dc[date_col], errors="coerce") if date_col else None
    out["status"]       = dc[status_col].astype(str) if status_col else None
    out["raw_row"]      = dc.apply(lambda r: json.dumps(r.where(r.notna(), None).to_dict()), axis=1)
    return out.reset_index(drop=True)


def write_to_duckdb(df: pd.DataFrame) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    df = df.copy()
    df["fetch_timestamp"] = ts

    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ferc_interconnection_datacenter (
                iso             VARCHAR,
                project_name    VARCHAR,
                capacity_mw     DOUBLE,
                state           VARCHAR,
                queue_date      DATE,
                status          VARCHAR,
                raw_row         VARCHAR,
                fetch_timestamp TIMESTAMP
            )
        """)
        conn.register("incoming", df)
        # Full refresh per ISO (idempotent: delete existing rows for fetched ISOs)
        for iso in df["iso"].unique():
            conn.execute(
                "DELETE FROM ferc_interconnection_datacenter WHERE iso = ?", [iso]
            )
        conn.execute("""
            INSERT INTO ferc_interconnection_datacenter
            SELECT iso, project_name,
                   TRY_CAST(capacity_mw AS DOUBLE),
                   state,
                   TRY_CAST(queue_date AS DATE),
                   status, raw_row,
                   fetch_timestamp::TIMESTAMP
            FROM incoming
        """)
        n = conn.execute(
            "SELECT COUNT(*) FROM ferc_interconnection_datacenter"
        ).fetchone()[0]
    print(f"   ✅ ferc_interconnection_datacenter: {n} total rows in warehouse")


def main():
    results = []
    fetch_log = {}

    for iso, cfg in ISO_SOURCES.items():
        print(f"\n🔌 {iso}  ({cfg['note']})")
        data = _download(iso, cfg["url"])
        if data is None:
            fetch_log[iso] = "FAILED — manual download required"
            continue
        df = parse_queue(iso, data, cfg)
        if df is not None and not df.empty:
            results.append(df)
            fetch_log[iso] = f"OK — {len(df)} datacenter rows"
        else:
            fetch_log[iso] = "OK — 0 datacenter keyword matches"

    print("\n--- Fetch summary ---")
    for iso, status in fetch_log.items():
        print(f"  {iso:8s}: {status}")

    if results:
        combined = pd.concat(results, ignore_index=True)
        write_to_duckdb(combined)
        print(f"\n✅ FERC ingest complete — {len(combined)} datacenter queue rows across "
              f"{combined['iso'].nunique()} ISOs")
        print("\n   Capacity by ISO (MW):")
        summary = (combined.groupby("iso")["capacity_mw"]
                   .agg(["count", "sum"])
                   .rename(columns={"count": "projects", "sum": "total_mw"}))
        print(summary.to_string())
    else:
        print("\n⚠️  No datacenter queue rows collected. "
              "Check the fetch log above for ISOs requiring manual download.")
        print("   Save queue Excel files to data/raw/ferc_queues/<ISO>_queue.xlsx "
              "and re-run to parse cached files.")


if __name__ == "__main__":
    main()
