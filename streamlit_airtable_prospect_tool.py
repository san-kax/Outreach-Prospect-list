# streamlit_airtable_prospect_tool.py

import os
import re
import time
import logging
from urllib.parse import urlparse
from datetime import datetime
from typing import Iterable

import streamlit as st
import pandas as pd
from pyairtable import Api

# ---------------- Config ----------------
logging.basicConfig(level=logging.INFO)

# Accept either AIRTABLE_TOKEN or airtable_token from secrets/env
AIRTABLE_TOKEN = (
    st.secrets.get("AIRTABLE_TOKEN")
    or st.secrets.get("airtable_token")
    or os.getenv("AIRTABLE_TOKEN")
)
if not AIRTABLE_TOKEN:
    st.error("❌ Missing Airtable token. Add AIRTABLE_TOKEN (or airtable_token) in Streamlit secrets or env.")
    st.stop()

api = Api(AIRTABLE_TOKEN)

# Bases/Tables (latest IDs)
# NOTE: Prospect-Data-1 is first so it is the ALWAYS-checked table and the push target.
SOURCES = [
    {"label": "Prospect-Data-1",   "base_id": "appVyliM5boVyoBhf", "table_id": "tbliCOQZY9RICLsLP"},  # ALWAYS checked + push target
    {"label": "Prospect-Data",     "base_id": "appHdhjsWVRxaCvcR", "table_id": "tbliCOQZY9RICLsLP"},
    {"label": "GDC-Database",      "base_id": "appUoOvkqzJvyyMvC", "table_id": "tbliCOQZY9RICLsLP"},
    {"label": "WB-Database",       "base_id": "appueIgn44RaVH6ot", "table_id": "tbl3vMYv4RzKfuBf4"},
    {"label": "Freebets-Database", "base_id": "appFBasaCUkEKtvpV", "table_id": "tblmTREzfIswOuA0F"},
]

# Fixed push target: ALWAYS push to Prospect-Data-1 (first source)
PUSH_BASE_ID  = SOURCES[0]["base_id"]
PUSH_TABLE_ID = SOURCES[0]["table_id"]
push_table = api.base(PUSH_BASE_ID).table(PUSH_TABLE_ID)

# --------------- Helpers ---------------
DOMAIN_RE = re.compile(r"^[a-z0-9.-]+$", re.IGNORECASE)

def normalize_domain(raw: str) -> str | None:
    """Lowercase, strip protocol/path/query/fragment, drop 'www.', IDN→punycode, basic validation."""
    import idna
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    if not s:
        return None
    if "://" in s:
        s = urlparse(s).netloc or s
    s = s.split("/")[0].split("?")[0].split("#")[0]
    s = s.rstrip(".")
    if s.startswith("www."):
        s = s[4:]
    if not s or "." not in s:
        return None
    try:
        s = idna.encode(s).decode("ascii")
    except Exception:
        return None
    if not DOMAIN_RE.match(s):
        return None
    return s

def chunked(iterable: Iterable, n: int = 10):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    """Robust reader for CSV/XLSX/XLS/XLSM with friendly errors."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file, dtype=str)
        elif name.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
            # Requires openpyxl
            return pd.read_excel(uploaded_file, dtype=str, engine="openpyxl")
        elif name.endswith(".xls"):
            # Old Excel format needs xlrd
            return pd.read_excel(uploaded_file, dtype=str, engine="xlrd")
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            st.stop()
    except ImportError as e:
        msg = str(e).lower()
        if "openpyxl" in msg:
            st.error("Excel reading requires **openpyxl**. Add `openpyxl>=3.1.2` to requirements.txt or upload a CSV.")
        elif "xlrd" in msg:
            st.error("Legacy .xls reading requires **xlrd**. Add `xlrd>=2.0.1` to requirements.txt or upload a CSV.")
        else:
            st.error(f"Missing dependency while reading your file: {e}")
        st.stop()

@st.cache_data(ttl=120)
def fetch_existing_domains(selected_sources: list[dict]) -> set[str]:
    """Read 'Domain' from all selected bases/tables and return a unified normalized set."""
    all_domains: set[str] = set()
    for src in selected_sources:
        table = api.base(src["base_id"]).table(src["table_id"])
        records = table.all(fields=["Domain"])  # pull only the needed field
        for r in records:
            d = normalize_domain(r.get("fields", {}).get("Domain", ""))
            if d:
                all_domains.add(d)
    return all_domains

def _escape_for_formula(val: str) -> str:
    # Airtable formula strings are single-quoted; escape any single quotes
    return val.replace("'", "\\'")

def domain_exists_in_prospect(domain_norm: str) -> bool:
    """Guard check in the push target (Prospect-Data-1)."""
    try:
        formula = f"LOWER({{Domain}}) = '{_escape_for_formula(domain_norm.lower())}'"
        recs = push_table.all(formula=formula, max_records=1, fields=["Domain"])
        return len(recs) > 0
    except Exception:
        # On any API error, be conservative to avoid duplicates.
        return True

# ---------------- UI ----------------
st.title("🔗 Prospect Filtering & Airtable Sync")
st.caption("Prospect-Data-1 is always checked for duplicates. You can include other databases too. Push goes to Prospect-Data-1 only. Duplicate-safe.")

st.subheader("👤 User")
user_name  = st.text_input("Your name:")
user_email = st.text_input("Your email:")
if not user_name or not user_email:
    st.warning("⚠️ Please provide your name and email to continue.")
    st.stop()

# Prospect-Data-1 is ALWAYS included in duplicate check
prospect_source = SOURCES[0]
other_sources = SOURCES[1:]

# Let user choose *additional* sources to check
options = [f'{s["label"]} ({s["base_id"]}:{s["table_id"]})' for s in other_sources]
selected_labels = st.multiselect(
    "Select additional Airtable sources to check for existing domains (Prospect-Data-1 is always checked)",
    options=options,
    default=options,
    help="These extra tables will also be scanned to remove duplicates before pushing."
)
active_sources = [prospect_source] + [s for s, lbl in zip(other_sources, options) if lbl in selected_labels]

uploaded_file = st.file_uploader(
    "Upload your prospect domains (CSV/Excel)",
    type=["csv", "xlsx", "xls", "xlsm"]
)
if not uploaded_file:
    st.stop()

# ---------- Read and normalize upload ----------
with st.spinner("Reading file..."):
    df_new = read_uploaded_table(uploaded_file)

if "Domain" not in df_new.columns:
    st.error("Your file must have a 'Domain' column.")
    st.stop()

raw = df_new["Domain"].dropna().astype(str).tolist()
new_domains = {d for d in (normalize_domain(x) for x in raw) if d}

st.write(f"📥 Uploaded rows: **{len(raw)}** | After normalization: **{len(new_domains)}**")

# ---------- Initial dedupe across selected sources ----------
with st.spinner("Fetching existing domains from Airtable..."):
    existing = fetch_existing_domains(active_sources)
st.info(f"📚 Existing domains across selected sources: **{len(existing)}**")

new_to_outreach = sorted(d for d in new_domains if d not in existing)
st.success(f"✅ {len(new_to_outreach)} new domains currently safe to outreach (pre-push check).")

df_result = pd.DataFrame({"Domain": new_to_outreach})
st.dataframe(df_result, use_container_width=True)
st.download_button("⬇️ Download Prospects (CSV)", df_result.to_csv(index=False), "prospects.csv")

# ---------- Push to Prospect-Data-1 only (duplicate-safe) ----------
if new_to_outreach:
    if "pushed" not in st.session_state:
        st.session_state.pushed = False
    disabled = st.session_state.pushed

    st.write(f"Target for push → **Prospect-Data-1** (`{PUSH_BASE_ID}:{PUSH_TABLE_ID}`)")
    if st.button(f"📤 Push {len(new_to_outreach)} Prospects to Airtable (duplicate-safe)", disabled=disabled):
        with st.spinner("Re-checking latest records and creating new ones..."):
            # Just-in-time recheck to shrink race window
            latest_existing = fetch_existing_domains(active_sources)
            to_push = [d for d in new_to_outreach if d not in latest_existing]

            created = 0
            skipped = 0
            errors  = 0
            date_str = datetime.now().strftime("%Y-%m-%d")

            for d in to_push:
                # Final guard in Prospect-Data-1 itself (handles true concurrency)
                if domain_exists_in_prospect(d):
                    skipped += 1
                    continue
                try:
                    push_table.create({
                        "Domain": d,
                        "Date": date_str,
                        "Added By Name": user_name,
                        "Added By Email": user_email
                    })
                    created += 1
                    time.sleep(0.05)  # tiny pacing helps with rate limits
                except Exception:
                    errors += 1

            st.session_state.pushed = True
            st.success(f"✅ Created: {created}  •  ⏭️ Skipped (already existed): {skipped}  •  ⚠️ Errors: {errors}")
