import os
import re
import time
import logging
from urllib.parse import urlparse
from datetime import datetime, timedelta
from typing import Iterable

# Try to import dateutil for better date parsing, but fallback to basic parsing if not available
try:
    from dateutil import parser as date_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False

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
SOURCES = [
    {"label": "Prospect-Data-1",     "base_id": "appVyIiM5boVyoBhf", "table_id": "tbliCOQZY9RICLsLP", "is_disavow": False},  # ALWAYS checked + push target
    {"label": "Prospect-Data",       "base_id": "appHdhjsWVRxaCvcR", "table_id": "tbliCOQZY9RICLsLP", "is_disavow": False},
    {"label": "GDC-Database",        "base_id": "appUoOvkqzJvyyMvC", "table_id": "tbliCOQZY9RICLsLP", "is_disavow": False},
    {"label": "WB-Database",         "base_id": "appueIgn44RaVH6ot", "table_id": "tbl3vMYv4RzKfuBf4", "is_disavow": False},
    {"label": "Freebets-Database",   "base_id": "appFBasaCUkEKtvpV", "table_id": "tblmTREzfIswOuA0F", "is_disavow": False},
    {"label": "GDC-Disavow-List",    "base_id": "appJTJQwjHRaAyLkw", "table_id": "tbliCOQZY9RICLsLP", "is_disavow": True},  # ALWAYS exclude - disavow list
    {"label": "GDC-Disavow-List-1",  "base_id": "appEEpV8mgLcBMQLE", "table_id": "tbliCOQZY9RICLsLP", "is_disavow": True},  # ALWAYS exclude - disavow list
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

def parse_date(date_str: str) -> datetime | None:
    """Parse date string from Airtable (handles various formats)."""
    if not date_str:
        return None
    try:
        if isinstance(date_str, str):
            date_str = date_str.strip()
            
            # Try dateutil parser first if available (handles most formats)
            if HAS_DATEUTIL:
                try:
                    return date_parser.parse(date_str, fuzzy=False)
                except Exception:
                    pass
            
            # Fallback to datetime.strptime for common formats
            # Common Airtable date formats
            formats = [
                "%Y-%m-%d",              # 2024-11-25
                "%m/%d/%Y",              # 11/25/2024
                "%d/%m/%Y",              # 25/11/2024
                "%Y-%m-%d %H:%M:%S",     # 2024-11-25 10:30:00
                "%m/%d/%Y %H:%M",        # 11/25/2024 10:30
                "%d/%m/%Y %H:%M",        # 25/11/2024 10:30
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
        return None
    except Exception:
        pass
    return None

def find_date_field(record_fields: dict) -> str | None:
    """Find date field in record, checking common field names."""
    # Common date field names in Airtable
    date_field_names = [
        "Date", "date", "Added Date", "Added date", "Publication Date", 
        "Publication date", "Created Date", "Created date", "First seen",
        "Last seen", "Date Added", "Date added"
    ]
    for field_name in date_field_names:
        if field_name in record_fields:
            return record_fields[field_name]
    return None

def is_domain_safe_to_reuse(record_fields: dict, months_threshold: int = 12) -> bool:
    """Check if domain is older than threshold months and can be reused."""
    date_field = find_date_field(record_fields)
    if not date_field:
        # If no date, consider it as old (safe to reuse) - conservative approach
        return True
    
    parsed_date = parse_date(str(date_field))
    if not parsed_date:
        # If date parsing fails, consider it old (safe to reuse)
        return True
    
    # Calculate if domain is older than threshold months
    threshold_date = datetime.now() - timedelta(days=months_threshold * 30)
    return parsed_date < threshold_date

@st.cache_data(ttl=120)
def fetch_existing_domains(selected_sources: list[dict], show_progress: bool = False, exclude_old_domains: bool = True, months_threshold: int = 12) -> tuple[set[str], dict[str, int], dict[str, int]]:
    """Read 'Domain' from all selected bases/tables and return a unified normalized set.
    
    Args:
        selected_sources: List of source dictionaries with base_id, table_id, label
        show_progress: Whether to log progress
        exclude_old_domains: If True, exclude domains older than months_threshold (SAFE to reuse)
        months_threshold: Number of months after which a domain is considered SAFE to reuse
    
    Returns:
        tuple: (all_domains_set, source_counts_dict, safe_domains_dict) where:
            - all_domains_set: Domains that should be excluded (not safe to reuse)
            - source_counts_dict: Total domains per source
            - safe_domains_dict: Count of SAFE (reusable) domains per source
    """
    all_domains: set[str] = set()
    source_counts: dict[str, int] = {}
    safe_domains: dict[str, int] = {}
    
    for src in selected_sources:
        try:
            table = api.base(src["base_id"]).table(src["table_id"])
            # Fetch ALL fields - don't restrict to specific field names
            # This ensures we get all records even if date field names differ
            records = table.all()  # Get all records with all fields
            count = 0
            safe_count = 0
            
            is_disavow_list = src.get("is_disavow", False)
            
            # Debug: Log available fields for first record (only once per source)
            fields_logged = False
            
            for r in records:
                fields = r.get("fields", {})
                
                # Debug: Log field names for first record to help diagnose issues
                if not fields_logged and fields:
                    available_fields = list(fields.keys())
                    logging.info(f"Available fields in {src['label']}: {available_fields[:10]}")  # Log first 10 fields
                    fields_logged = True
                
                # Get Domain field (try common variations)
                # Check multiple possible field names
                domain_value = (
                    fields.get("Domain") or 
                    fields.get("domain") or 
                    fields.get("A Domain") or 
                    fields.get("Live Link") or
                    fields.get("Referring page URL") or
                    # Try to find any field that might contain a domain/URL
                    next((v for k, v in fields.items() if isinstance(v, str) and ("http" in v.lower() or ("." in v and len(v.split(".")) >= 2)) and len(v) < 200), None)
                )
                
                # If domain_value is a URL, normalize_domain will extract the domain
                d = normalize_domain(domain_value)
                if d:
                    count += 1
                    
                    # Disavow lists: ALWAYS exclude, regardless of age
                    if is_disavow_list:
                        all_domains.add(d)  # Always exclude from safe to outreach
                        # Don't count as SAFE for disavow lists
                    else:
                        # Regular sources: apply 12-month rule
                        # Check if domain is old enough to be SAFE (reusable)
                        if exclude_old_domains and is_domain_safe_to_reuse(fields, months_threshold):
                            safe_count += 1
                            # Don't add to all_domains - it's SAFE to reuse
                        else:
                            # Domain is still "active" (less than 12 months old) - exclude it
                            all_domains.add(d)
            
            source_counts[src["label"]] = count
            safe_domains[src["label"]] = safe_count
            if show_progress:
                active = count - safe_count
                logging.info(f"✓ Fetched {count:,} domains from {src['label']} ({src['base_id']}) - {active:,} active, {safe_count:,} SAFE (12+ months old)")
        except Exception as e:
            logging.error(f"✗ Error fetching from {src['label']}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            source_counts[src["label"]] = 0
            safe_domains[src["label"]] = 0
    
    return all_domains, source_counts, safe_domains

def _escape_for_formula(val: str) -> str:
    """Escape special characters for Airtable formula strings."""
    # Escape backslashes first, then single quotes
    return val.replace("\\", "\\\\").replace("'", "\\'")

def domain_exists_in_prospect(domain_norm: str) -> bool:
    """Guard check in the push target (Prospect-Data-1)."""
    try:
        formula = f"LOWER({{Domain}}) = '{_escape_for_formula(domain_norm.lower())}'"
        recs = push_table.all(formula=formula, max_records=1, fields=["Domain"])
        return len(recs) > 0
    except Exception as e:
        logging.error(f"Error checking domain existence for {domain_norm}: {e}")
        # Return True to be safe (skip rather than duplicate)
        return True

# ---------------- UI ----------------
st.title("🔗 Prospect Filtering & Airtable Sync")
st.caption("Prospect-Data-1 is always checked for duplicates. You can include other databases too. Push goes to Prospect-Data-1 only. Duplicate-safe.")

st.subheader("👤 User")
user_name  = st.text_input("Your name:")
user_email = st.text_input("Your email:")

# Basic email validation
def is_valid_email(email: str) -> bool:
    """Basic email format validation."""
    if not email or "@" not in email:
        return False
    parts = email.split("@")
    if len(parts) != 2:
        return False
    if not parts[0] or not parts[1] or "." not in parts[1]:
        return False
    return True

if not user_name or not user_email:
    st.warning("⚠️ Please provide your name and email to continue.")
    st.stop()

if not is_valid_email(user_email):
    st.error("❌ Please provide a valid email address.")
    st.stop()

# Prospect-Data-1 is ALWAYS included in duplicate check
prospect_source = SOURCES[0]
other_sources = SOURCES[1:]

# Let user choose *additional* sources to check
st.write("**📊 Deduplication Sources:**")
st.markdown(f"🔒 **Always checked:** `{prospect_source['label']}` (cannot be deselected)")

# Create cleaner labels for multiselect
options = [s["label"] for s in other_sources]
option_to_source = {s["label"]: s for s in other_sources}

selected_labels = st.multiselect(
    "Select additional Airtable sources to check for duplicates:",
    options=options,
    default=options,
    help="These databases will be scanned along with Prospect-Data-1 to remove duplicates before pushing."
)

active_sources = [prospect_source] + [option_to_source[label] for label in selected_labels]

# Show which sources are active
st.info(f"**Active sources ({len(active_sources)}):** " + 
        " • ".join([f"`{src['label']}`" for src in active_sources]))

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
st.write("**⏰ 12-Month Reuse Policy:**")
st.caption("Domains older than 12 months are marked as SAFE and can be reused (excluded from duplicate check). **Disavow lists are ALWAYS excluded regardless of age.**")

with st.spinner("Fetching existing domains from Airtable..."):
    try:
        existing, source_counts, safe_domains = fetch_existing_domains(
            active_sources, 
            show_progress=True, 
            exclude_old_domains=True, 
            months_threshold=12
        )
    except Exception as e:
        st.error(f"❌ Error fetching domains: {e}")
        logging.error(f"Error in fetch_existing_domains: {e}")
        import traceback
        logging.error(traceback.format_exc())
        existing = set()
        source_counts = {src["label"]: 0 for src in active_sources}
        safe_domains = {src["label"]: 0 for src in active_sources}

# Calculate totals
total_domains = sum(source_counts.values())
total_safe = sum(safe_domains.values())
total_active = total_domains - total_safe

# Display detailed breakdown
st.info(f"📚 **Total domains:** **{total_domains:,}** | **Active (<12 months):** **{total_active:,}** | **🟢 SAFE (12+ months, reusable):** **{total_safe:,}**")
st.info(f"🔒 **Domains excluded from upload (active duplicates):** **{len(existing):,}**")

with st.expander("📋 View detailed breakdown by source"):
    for src in active_sources:
        count = source_counts.get(src["label"], 0)
        safe = safe_domains.get(src["label"], 0)
        active = count - safe
        is_disavow = src.get("is_disavow", False)
        disavow_note = " 🚫 DISAVOW (always excluded)" if is_disavow else ""
        st.write(f"  • **{src['label']}**: {count:,} total ({active:,} active, {safe:,} SAFE){disavow_note}")

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
            latest_existing, _, _ = fetch_existing_domains(
                active_sources, 
                exclude_old_domains=True, 
                months_threshold=12
            )
            to_push = [d for d in new_to_outreach if d not in latest_existing]
            
            if len(to_push) < len(new_to_outreach):
                st.warning(f"⚠️ {len(new_to_outreach) - len(to_push)} domains were added by another process. Only {len(to_push)} will be pushed.")

            created = 0
            skipped = 0
            errors  = 0
            error_details = []
            date_str = datetime.now().strftime("%Y-%m-%d")

            progress_bar = st.progress(0)
            total = len(to_push)
            
            for idx, d in enumerate(to_push):
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
                    time.sleep(0.05)
                except Exception as e:
                    errors += 1
                    error_details.append(f"{d}: {str(e)}")
                    logging.error(f"Error creating record for {d}: {e}")
                
                # Update progress
                if total > 0:
                    progress_bar.progress((idx + 1) / total)

            progress_bar.empty()
            st.session_state.pushed = True
            
            # Show results
            st.success(f"✅ **Created:** {created}  •  ⏭️ **Skipped (already existed):** {skipped}  •  ⚠️ **Errors:** {errors}")
            
            if errors > 0 and error_details:
                with st.expander("⚠️ View error details"):
                    for err in error_details[:10]:  # Show first 10 errors
                        st.text(err)
                    if len(error_details) > 10:
                        st.text(f"... and {len(error_details) - 10} more errors")
