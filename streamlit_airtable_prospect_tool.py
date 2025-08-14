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
from pyairtable.api.types import AirtableError

# ---------------- Config ----------------
logging.basicConfig(level=logging.INFO)

AIRTABLE_TOKEN = st.secrets.get("AIRTABLE_TOKEN") or os.getenv("AIRTABLE_TOKEN")
if not AIRTABLE_TOKEN:
    st.error("❌ Missing Airtable token. Add AIRTABLE_TOKEN in Streamlit secrets or env.")
    st.stop()

api = Api(AIRTABLE_TOKEN)

# Your 4 bases/tables (from the URLs you shared)
SOURCES = [
    {"label": "Base A", "base_id": "appHdhjsWVRxaCvcR", "table_id": "tbliCOQZY9RICLsLP"},
    {"label": "Base B", "base_id": "apprZEmIUaqjzuurQ", "table_id": "tbliCOQZY9RICLsLP"},
    {"label": "Base C", "base_id": "appueIgn44RaVH6ot", "table_id": "tbl3vMYv4RzKfuBf4"},
    {"label": "Base D", "base_id": "appFBasaCUkEKtvpV", "table_id": "tblmTREzfIswOuA0F"},
]

# --------------- Helpers ---------------
DOMAIN_RE = re.compile(r"^[a-z0-9.-]+$", re.IGNORECASE)

def normalize_domain(raw: str) -> str | None:
    """Lowercase, strip proto/path, drop 'www.', IDN->punycode, sanity checks."""
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
    except idna.IDNAError:
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

def batch_create_domains(table, domains, user_name, user_email, date_str):
    created = 0
    for batch in chunked(domains, 10):
        records = [{"fields": {
            "Domain": d,
            "Date": date_str,
            "Added By Name": user_name,
            "Added By Email": user_email
        }} for d in batch]
        for attempt in range(3):
            try:
                table.batch_create(records)
                created += len(records)
                break
            except AirtableError as e:
                msg = str(e).lower()
                if ("rate limit" in msg or "429" in msg) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise
    return created

@st.cache_data(ttl=120)
def fetch_existing_domains(selected_sources: list[dict]) -> set[str]:
    all_domains: set[str] = set()
    for src in selected_sources:
        table = api.base(src["base_id"]).table(src["table_id"])
        records = table.all(fields=["Domain"])  # faster than .all()
        for r in records:
            d = normalize_domain(r.get("fields", {}).get("Domain", ""))
            if d:
                all_domains.add(d)
    return all_domains

# ---------------- UI ----------------
st.title("🔗 Prospect Filtering & Airtable Sync (4 databases)")

st.subheader("👤 User")
user_name  = st.text_input("Your name:")
user_email = st.text_input("Your email:")
if not user_name or not user_email:
    st.warning("⚠️ Please provide your name and email to continue.")
    st.stop()

# Choose which sources to check for duplicates
options = [f'{s["label"]} ({s["base_id"]}:{s["table_id"]})' for s in SOURCES]
selected_labels = st.multiselect(
    "Select Airtable sources to check for existing domains",
    options=options,
    default=options
)
active_sources = [s for s, lbl in zip(SOURCES, options) if lbl in selected_labels]

# Choose where to push
push_label = st.selectbox(
    "Choose the Airtable table to PUSH new prospects to",
    options=options,
    index=0
)
push_src = [s for s, lbl in zip(SOURCES, options) if lbl == push_label][0]
push_table = api.base(push_src["base_id"]).table(push_src["table_id"])

uploaded_file = st.file_uploader("Upload your prospect domains (CSV/Excel)", type=["csv", "xlsx"])
if not uploaded_file:
    st.stop()

# Read file
with st.spinner("Reading file..."):
    if uploaded_file.name.endswith(".csv"):
        df_new = pd.read_csv(uploaded_file, dtype=str)
    else:
        df_new = pd.read_excel(uploaded_file, dtype=str)

if "Domain" not in df_new.columns:
    st.error("Your file must have a 'Domain' column.")
    st.stop()

# Normalize uploaded domains
raw = df_new["Domain"].dropna().astype(str).tolist()
new_domains = {d for d in (normalize_domain(x) for x in raw) if d}
st.write(f"📥 Uploaded rows: **{len(raw)}** | After normalization: **{len(new_domains)}**")

# Pull existing domains from the selected sources
with st.spinner("Fetching existing domains from Airtable..."):
    existing = fetch_existing_domains(active_sources)
st.info(f"📚 Existing domains across selected sources: **{len(existing)}**")

# Diff
new_to_outreach = sorted(d for d in new_domains if d not in existing)
st.success(f"✅ {len(new_to_outreach)} new domains safe to outreach.")

df_result = pd.DataFrame({"Domain": new_to_outreach})
st.dataframe(df_result, use_container_width=True)
st.download_button("⬇️ Download Prospects (CSV)", df_result.to_csv(index=False), "prospects.csv")

# Push
if new_to_outreach:
    if "pushed" not in st.session_state:
        st.session_state.pushed = False
    disabled = st.session_state.pushed

    if st.button(f"📤 Push {len(new_to_outreach)} Prospects to Airtable", disabled=disabled):
        with st.spinner("Creating records in Airtable..."):
            try:
                created = batch_create_domains(
                    push_table,
                    new_to_outreach,
                    user_name,
                    user_email,
                    datetime.now().strftime("%Y-%m-%d"),
                )
                st.session_state.pushed = True
                st.success(
                    f"🎉 Added {created} new domains to `{push_src['label']}` "
                    f"({push_src['base_id']}:{push_src['table_id']})."
                )
            except Exception as e:
                st.error(f"❌ Push failed: {e}")
