import os
import re
import time
import logging
import traceback
from urllib.parse import urlparse
from datetime import datetime, timedelta
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import idna

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

# Helper function to test base access with detailed diagnostics
def test_base_access(base_id: str, table_id: str) -> tuple[bool, str]:
    """Test if we can access a specific base/table. Returns (success, message)."""
    try:
        base = api.base(base_id)
        table = base.table(table_id)

        # Test 1: Try to get schema (requires schema.bases:read)
        schema_ok = False
        try:
            # Try accessing base metadata
            base_info = base.schema()
            schema_ok = True
        except Exception as schema_err:
            schema_err_str = str(schema_err)
            if "403" in schema_err_str:
                return False, f"403 Forbidden - Cannot access base schema. Token may need 'schema.bases:read' scope or workspace access."

        # Test 2: Try to get just one record (requires data.records:read)
        try:
            records = table.all(max_records=1)
            return True, "✅ Access OK - Can read schema and data"
        except Exception as data_error:
            error_str = str(data_error)
            if "403" in error_str or "Forbidden" in error_str:
                if schema_ok:
                    return False, f"❌ 403 Forbidden - Can see base schema but cannot read records. Token may need 'data.records:read' scope."
                else:
                    return False, f"❌ 403 Forbidden - Cannot access base. Check: 1) Token scopes 2) Workspace permissions 3) Base is in correct workspace"
            return False, f"Error: {error_str[:150]}"
    except Exception as e:
        error_str = str(e)
        if "403" in error_str or "Forbidden" in error_str:
            return False, f"❌ 403 Forbidden - Token may not have access to base {base_id}"
        return False, f"Error: {error_str[:150]}"

# ---- Verticals: each has its own Prospect-Data push target ----
VERTICALS = {
    "GDC": {
        "prospect_base_id": "appVyIiM5boVyoBhf",
        "prospect_table_id": "tbliCOQZY9RICLsLP",
        "prospect_label": "Prospect-Data-GDC-1",
        "extra_prospect": {"label": "Prospect-Data-GDC", "base_id": "appHdhjsWVRxaCvcR", "table_id": "tbliCOQZY9RICLsLP"},
    },
    "WhichBingo": {
        "prospect_base_id": "appphIl2Iq8kloRGD",
        "prospect_table_id": "tbliCOQZY9RICLsLP",
        "prospect_label": "Prospect-Data-WhichBingo",
    },
    "BonusFinder": {
        "prospect_base_id": "app7LTnZSYutwKzsx",
        "prospect_table_id": "tbliCOQZY9RICLsLP",
        "prospect_label": "Prospect-Data-BonusFinder",
    },
    "Freebets": {
        "prospect_base_id": "appzbw2BJVm5QXCAa",
        "prospect_table_id": "tbliCOQZY9RICLsLP",
        "prospect_label": "Prospect-Data-Freebets",
    },
    "Bookies": {
        "prospect_base_id": "appZfavfEMOpPbqiP",
        "prospect_table_id": "tbliCOQZY9RICLsLP",
        "prospect_label": "Prospect-Data-Bookies",
    },
    "Casinos": {
        "prospect_base_id": "appO5ta4j5rUaG9XL",
        "prospect_table_id": "tbliCOQZY9RICLsLP",
        "prospect_label": "Prospect-Data-Casinos",
    },
    "Rotowire": {
        "prospect_base_id": "appwEdvjcFpq4qiHj",
        "prospect_table_id": "tbliCOQZY9RICLsLP",
        "prospect_label": "Prospect-Data-Rotowire",
    },
}

# ---- Shared deduplication sources (checked across all verticals) ----
SHARED_SOURCES = [
    {"label": "GDC-Database",        "base_id": "appUoOvkqzJvyyMvC", "table_id": "tbliCOQZY9RICLsLP", "is_disavow": False},
    {"label": "WB-Database",         "base_id": "appueIgn44RaVH6ot", "table_id": "tbl3vMYv4RzKfuBf4", "is_disavow": False},
    {"label": "Freebets-Database",   "base_id": "appFBasaCUkEKtvpV", "table_id": "tblmTREzfIswOuA0F", "is_disavow": False},
    {"label": "GDC-Disavow-List",    "base_id": "appJTJQwjHRaAyLkw", "table_id": "tbliCOQZY9RICLsLP", "is_disavow": True},
    {"label": "GDC-Disavow-List-1",  "base_id": "appEEpV8mgLcBMQLE", "table_id": "tbliCOQZY9RICLsLP", "is_disavow": True},
    {"label": "Local States Vertical Live Links", "base_id": "app08yUTcPhJVPxCI", "table_id": "tbllmyX2xNVXMEEnc", "is_disavow": False},
    {"label": "Sports Vertical Bookies.com and Rotowire", "base_id": "appDFsy6RWw5TRNH6", "table_id": "tbl8whN06WyCOo5uk", "is_disavow": False},
    {"label": "Casinos-Links",       "base_id": "appay75NrffUxBMbM", "table_id": "tblx8ZGluvQ9cWdXh", "is_disavow": False},
]

# --------------- Helpers ---------------
DOMAIN_RE = re.compile(r"^[a-z0-9.-]+$", re.IGNORECASE)

def normalize_domain(raw: str) -> str | None:
    """Lowercase, strip protocol/path/query/fragment, drop 'www.', IDN->punycode, basic validation."""
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

    # Handle timezone-aware vs timezone-naive datetime comparison
    # Make both timezone-naive for comparison
    if parsed_date.tzinfo is not None:
        # Remove timezone info for comparison
        parsed_date = parsed_date.replace(tzinfo=None)
    if threshold_date.tzinfo is not None:
        threshold_date = threshold_date.replace(tzinfo=None)

    return parsed_date < threshold_date

def fetch_single_source(src: dict, exclude_old_domains: bool, months_threshold: int, return_domain_dates: bool = False, prospect_data_labels: set[str] | None = None) -> tuple[str, set[str], int, int, str | None, dict[str, datetime] | None]:
    """Fetch domains from a single Airtable source. Returns (label, domains_set, count, safe_count, field_used, domain_dates_dict).

    Args:
        return_domain_dates: If True, also return a dict mapping domain -> date for Prospect-Data sources
        prospect_data_labels: Set of labels that are considered "Prospect-Data" sources for this vertical
    """
    try:
        table = api.base(src["base_id"]).table(src["table_id"])
        is_disavow_list = src.get("is_disavow", False)
        _prospect_labels = prospect_data_labels or set()
        is_prospect_data_source = src["label"] in _prospect_labels

        # Try to fetch only needed fields for better performance
        possible_domain_fields = ["Domain", "domain", "A Domain", "Live Link", "Referring page URL"]
        possible_date_fields = ["Date", "date", "Added Date", "Publication Date", "Created Date", "First seen", "Last seen"]

        # First, try fetching with specific fields (faster)
        try:
            all_fields_to_fetch = list(set(possible_domain_fields + possible_date_fields))
            records = table.all(fields=all_fields_to_fetch)
        except Exception as e1:
            error_str = str(e1)
            # Check if it's a permission error (403)
            if "403" in error_str or "Forbidden" in error_str or "INVALID_PERMISSIONS" in error_str:
                logging.error(f"⚠ Permission denied for {src['label']} (base: {src['base_id']}, table: {src['table_id']})")
                logging.error(f"   Even though token shows access, API returns 403. Possible causes:")
                logging.error(f"   1. Token needs to be saved/refreshed in Airtable")
                logging.error(f"   2. Base/table IDs might be incorrect")
                logging.error(f"   3. Different token is being used than configured")
                raise  # Re-raise to be caught by outer handler
            # If that fails for other reasons, fetch all fields (slower but more reliable)
            try:
                records = table.all()
            except Exception as e2:
                error_str2 = str(e2)
                # If even fetching all fields fails, check if it's a permission error
                if "403" in error_str2 or "Forbidden" in error_str2 or "INVALID_PERMISSIONS" in error_str2:
                    logging.error(f"⚠ Permission denied for {src['label']} (base: {src['base_id']}, table: {src['table_id']})")
                    logging.error(f"   Even though token shows access, API returns 403. Possible causes:")
                    logging.error(f"   1. Token needs to be saved/refreshed in Airtable")
                    logging.error(f"   2. Base/table IDs might be incorrect")
                    logging.error(f"   3. Different token is being used than configured")
                raise

        domains_set: set[str] = set()
        domain_dates: dict[str, datetime] = {} if return_domain_dates and is_prospect_data_source else {}
        count = 0
        safe_count = 0
        domain_field_found = None
        fields_logged = False

        for r in records:
            fields = r.get("fields", {})

            # Debug: Log field names for first record
            if not fields_logged and fields:
                available_fields = list(fields.keys())
                logging.info(f"Available fields in {src['label']}: {available_fields}")
                fields_logged = True

            # Get Domain field (try common variations)
            domain_value = None

            # First, try exact field name matches
            for field_name in possible_domain_fields:
                if field_name in fields:
                    domain_value = fields[field_name]
                    if not domain_field_found:
                        domain_field_found = field_name
                    break

            # If no exact match, try URL fields
            if not domain_value:
                for field_name in ["Live Link", "Referring page URL", "URL", "url"]:
                    if field_name in fields:
                        url_value = fields[field_name]
                        if url_value:
                            domain_value = url_value
                            if not domain_field_found:
                                domain_field_found = field_name
                            break

            # Last resort: find any field that looks like a domain/URL
            if not domain_value:
                for k, v in fields.items():
                    if isinstance(v, str) and v.strip():
                        if "http" in v.lower():
                            domain_value = v
                            if not domain_field_found:
                                domain_field_found = k
                            break
                        elif "." in v and len(v.split(".")) >= 2 and len(v) < 200:
                            if not v.replace(".", "").replace("-", "").isdigit():
                                domain_value = v
                                if not domain_field_found:
                                    domain_field_found = k
                                break

            # Normalize and process domain
            d = normalize_domain(domain_value)
            if d:
                count += 1

                # Store date for Prospect-Data sources if requested
                if return_domain_dates and is_prospect_data_source:
                    date_field = find_date_field(fields)
                    if date_field:
                        parsed_date = parse_date(str(date_field))
                        if parsed_date:
                            # Handle timezone-aware dates
                            if parsed_date.tzinfo is not None:
                                parsed_date = parsed_date.replace(tzinfo=None)
                            domain_dates[d] = parsed_date

                # Disavow lists: ALWAYS exclude
                if is_disavow_list:
                    domains_set.add(d)
                else:
                    # Regular sources: apply 12-month rule
                    if exclude_old_domains and is_domain_safe_to_reuse(fields, months_threshold):
                        safe_count += 1
                    else:
                        domains_set.add(d)

        if count == 0 and fields_logged:
            logging.warning(f"⚠ {src['label']} returned 0 domains - available fields: {list(fields.keys())}")

        return (src["label"], domains_set, count, safe_count, domain_field_found, domain_dates if return_domain_dates and is_prospect_data_source else None)

    except Exception as e:
        error_str = str(e)
        # Check if it's a permission error (403)
        if "403" in error_str or "Forbidden" in error_str or "INVALID_PERMISSIONS" in error_str:
            logging.error(f"✗ Permission denied for {src['label']} (base: {src['base_id']}, table: {src['table_id']})")
            logging.error(f"   API returns 403 even though token shows access. Troubleshooting:")
            logging.error(f"   1. Verify token in Streamlit secrets matches the token you're editing")
            logging.error(f"   2. Try saving the token again in Airtable (even if already saved)")
            logging.error(f"   3. Verify base_id '{src['base_id']}' and table_id '{src['table_id']}' are correct")
            logging.error(f"   4. Check if token has 'data.records:read' scope enabled")
        else:
            logging.error(f"✗ Error fetching from {src['label']} ({src['base_id']}): {e}")
            logging.error(traceback.format_exc())
        return (src["label"], set(), 0, 0, None, None)

@st.cache_data(ttl=300)  # Increased cache time to 5 minutes
def fetch_existing_domains(selected_sources: list[dict], show_progress: bool = False, exclude_old_domains: bool = True, months_threshold: int = 12, return_source_mapping: bool = False, prospect_data_labels: tuple[str, ...] = ()) -> tuple[set[str], dict[str, int], dict[str, int], dict[str, set[str]] | None, dict[str, dict[str, datetime]] | None]:
    """Read 'Domain' from all selected bases/tables and return a unified normalized set.

    Uses parallel processing to fetch from multiple sources simultaneously for better performance.

    Args:
        selected_sources: List of source dictionaries with base_id, table_id, label
        show_progress: Whether to log progress
        exclude_old_domains: If True, exclude domains older than months_threshold (SAFE to reuse)
        months_threshold: Number of months after which a domain is considered SAFE to reuse
        return_source_mapping: If True, also return domain_to_sources mapping and domain_dates_by_source
        prospect_data_labels: Tuple of labels that are considered "Prospect-Data" sources for this vertical

    Returns:
        tuple: (all_domains_set, source_counts_dict, safe_domains_dict, domain_to_sources, domain_dates_by_source) where:
            - all_domains_set: Domains that should be excluded (not safe to reuse)
            - source_counts_dict: Total domains per source
            - safe_domains_dict: Count of SAFE (reusable) domains per source
            - domain_to_sources: Dict mapping domain -> set of source labels (only if return_source_mapping=True)
            - domain_dates_by_source: Dict mapping source label -> dict(domain -> date) (only if return_source_mapping=True)
    """
    all_domains: set[str] = set()
    source_counts: dict[str, int] = {}
    safe_domains: dict[str, int] = {}
    domain_to_sources: dict[str, set[str]] = {} if return_source_mapping else None
    domain_dates_by_source: dict[str, dict[str, datetime]] = {} if return_source_mapping else None
    _prospect_labels_set = set(prospect_data_labels)

    # Process sources in parallel for better performance
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_source = {
            executor.submit(fetch_single_source, src, exclude_old_domains, months_threshold, return_source_mapping, _prospect_labels_set): src
            for src in selected_sources
        }

        for future in as_completed(future_to_source):
            result = future.result()
            label, domains_set, count, safe_count, field_used, domain_dates = result
            all_domains.update(domains_set)
            source_counts[label] = count
            safe_domains[label] = safe_count

            # Track which domains come from which sources
            if return_source_mapping:
                for domain in domains_set:
                    if domain not in domain_to_sources:
                        domain_to_sources[domain] = set()
                    domain_to_sources[domain].add(label)

                # Store domain dates for Prospect-Data sources
                if domain_dates:
                    domain_dates_by_source[label] = domain_dates

            if show_progress:
                active = count - safe_count
                field_info = f" (using field: {field_used})" if field_used else ""
                logging.info(f"✓ Fetched {count:,} domains from {label} - {active:,} active, {safe_count:,} SAFE{field_info}")

    return all_domains, source_counts, safe_domains, domain_to_sources, domain_dates_by_source

def identify_reoutreach_candidates(
    user_domains: set[str],
    domain_to_sources: dict[str, set[str]],
    domain_dates_by_source: dict[str, dict[str, datetime]],
    months_threshold: int = 3,
    prospect_data_labels: set[str] | None = None,
    push_target_label: str | None = None
) -> set[str]:
    """Identify domains that are safe for re-outreach.

    A domain is a re-outreach candidate if:
    1. It's in the user's input list
    2. It's found ONLY in Prospect-Data sources for this vertical (not in any other sources)
    3. The date in those sources is older than months_threshold (default 3 months)

    Args:
        user_domains: Set of normalized domains from user's upload
        domain_to_sources: Dict mapping domain -> set of source labels where it appears
        domain_dates_by_source: Dict mapping source label -> dict(domain -> date)
        months_threshold: Number of months after which a domain is considered old enough for re-outreach
        prospect_data_labels: Set of labels considered "Prospect-Data" sources for this vertical
        push_target_label: The label of the push target (checked first for date)

    Returns:
        Set of domains that are safe for re-outreach
    """
    reoutreach_candidates: set[str] = set()
    _prospect_labels = prospect_data_labels or set()
    _push_label = push_target_label
    threshold_date = datetime.now() - timedelta(days=months_threshold * 30)

    for domain in user_domains:
        sources = domain_to_sources.get(domain, set())

        # Check if domain is only in Prospect-Data sources (and actually in at least one)
        if sources and sources.issubset(_prospect_labels) and sources.intersection(_prospect_labels):
            # Check if domain is older than threshold
            # Priority: Check push target first, then other prospect sources
            is_old_enough = False

            # Priority: Check push target first if domain exists there
            if _push_label and _push_label in sources:
                if _push_label in domain_dates_by_source:
                    source_dates = domain_dates_by_source[_push_label]
                    if domain in source_dates:
                        domain_date = source_dates[domain]
                        if domain_date < threshold_date:
                            is_old_enough = True
            else:
                # Check other prospect sources
                for plabel in _prospect_labels:
                    if plabel in sources and plabel in domain_dates_by_source:
                        source_dates = domain_dates_by_source[plabel]
                        if domain in source_dates:
                            domain_date = source_dates[domain]
                            if domain_date < threshold_date:
                                is_old_enough = True
                            break

            if is_old_enough:
                reoutreach_candidates.add(domain)

    return reoutreach_candidates

def _escape_for_formula(val: str) -> str:
    """Escape special characters for Airtable formula strings."""
    # Escape backslashes first, then single quotes
    return val.replace("\\", "\\\\").replace("'", "\\'")

def domain_exists_in_prospect(domain_norm: str, push_table_ref) -> tuple[bool, str | None]:
    """Check if domain exists in the push target. Returns (exists, record_id)."""
    try:
        formula = f"LOWER({{Domain}}) = '{_escape_for_formula(domain_norm.lower())}'"
        recs = push_table_ref.all(formula=formula, max_records=1, fields=["Domain"])
        if len(recs) > 0:
            return True, recs[0].get("id")
        return False, None
    except Exception as e:
        logging.error(f"Error checking domain existence for {domain_norm}: {e}")
        # Return True to be safe (skip rather than duplicate)
        return True, None

# ---------------- UI ----------------
st.title("🔗 Prospect Filtering & Airtable Sync")

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

# ---------- Vertical selection ----------
st.subheader("🏢 Vertical")
selected_vertical = st.selectbox(
    "Select your vertical:",
    list(VERTICALS.keys()),
    help="Each vertical pushes prospects to its own Prospect-Data base."
)
vertical_config = VERTICALS[selected_vertical]

# Dynamic push target based on selected vertical
PUSH_BASE_ID = vertical_config["prospect_base_id"]
PUSH_TABLE_ID = vertical_config["prospect_table_id"]
push_table = api.base(PUSH_BASE_ID).table(PUSH_TABLE_ID)
push_target_label = vertical_config["prospect_label"]

# Build the prospect source (always checked — the push target itself)
prospect_source = {
    "label": vertical_config["prospect_label"],
    "base_id": vertical_config["prospect_base_id"],
    "table_id": vertical_config["prospect_table_id"],
    "is_disavow": False,
}

# Build the set of "prospect data labels" for re-outreach logic
prospect_data_labels = {vertical_config["prospect_label"]}

# Add extra prospect source if it exists (e.g., GDC has Prospect-Data-GDC as secondary)
extra_prospect_sources = []
if "extra_prospect" in vertical_config:
    extra = dict(vertical_config["extra_prospect"])  # copy to avoid mutating config
    extra["is_disavow"] = False
    extra_prospect_sources = [extra]
    prospect_data_labels.add(extra["label"])

# Include ALL other verticals' prospect bases as dedup sources
other_prospect_sources = []
for vname, vconfig in VERTICALS.items():
    if vname != selected_vertical:
        other_prospect_sources.append({
            "label": vconfig["prospect_label"],
            "base_id": vconfig["prospect_base_id"],
            "table_id": vconfig["prospect_table_id"],
            "is_disavow": False,
        })

# Combine: extra prospect sources + other verticals + shared sources
other_sources = extra_prospect_sources + other_prospect_sources + SHARED_SOURCES

st.caption(
    f"`{push_target_label}` is always checked for duplicates. "
    f"You can include other databases too. Push goes to `{push_target_label}` only. Duplicate-safe."
)

# ---------- Source selection ----------
st.write("**📊 Deduplication Sources:**")
st.markdown(f"🔒 **Always checked:** `{prospect_source['label']}` (cannot be deselected)")

# Create cleaner labels for multiselect
options = [s["label"] for s in other_sources]
option_to_source = {s["label"]: s for s in other_sources}

selected_labels = st.multiselect(
    "Select additional Airtable sources to check for duplicates:",
    options=options,
    default=options,
    help=f"These databases will be scanned along with {push_target_label} to remove duplicates before pushing."
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

st.write("**🔄 3-Month Re-Outreach Policy:**")
st.caption(f"Domains found ONLY in {push_target_label} prospect sources that are older than 3 months and not in any other sources are safe for re-outreach.")

# Convert prospect_data_labels to tuple for caching (sets are unhashable)
_prospect_labels_tuple = tuple(sorted(prospect_data_labels))

with st.spinner("Fetching existing domains from Airtable..."):
    try:
        existing, source_counts, safe_domains, domain_to_sources, domain_dates_by_source = fetch_existing_domains(
            active_sources,
            show_progress=True,
            exclude_old_domains=True,
            months_threshold=12,
            return_source_mapping=True,
            prospect_data_labels=_prospect_labels_tuple
        )
    except Exception as e:
        st.error(f"❌ Error fetching domains: {e}")
        logging.error(f"Error in fetch_existing_domains: {e}")
        logging.error(traceback.format_exc())
        existing = set()
        source_counts = {src["label"]: 0 for src in active_sources}
        safe_domains = {src["label"]: 0 for src in active_sources}
        domain_to_sources = {}
        domain_dates_by_source = {}

# Identify re-outreach candidates (domains in this vertical's Prospect-Data sources only, >3 months old)
reoutreach_candidates = set()
if domain_to_sources is not None and domain_dates_by_source is not None and domain_to_sources and domain_dates_by_source:
    reoutreach_candidates = identify_reoutreach_candidates(
        new_domains,
        domain_to_sources,
        domain_dates_by_source,
        months_threshold=3,
        prospect_data_labels=prospect_data_labels,
        push_target_label=push_target_label
    )
    # Remove re-outreach candidates from existing set (they're safe to outreach again)
    existing = existing - reoutreach_candidates

# Calculate totals
total_domains = sum(source_counts.values())
total_safe = sum(safe_domains.values())
total_active = total_domains - total_safe

# Display detailed breakdown
st.info(f"📚 **Total domains:** **{total_domains:,}** | **Active (<12 months):** **{total_active:,}** | **🟢 SAFE (12+ months, reusable):** **{total_safe:,}**")
st.info(f"🔒 **Domains excluded from upload (active duplicates):** **{len(existing):,}**")
if reoutreach_candidates:
    st.success(f"🔄 **Re-outreach candidates (3+ months old, only in {selected_vertical} Prospect-Data sources):** **{len(reoutreach_candidates):,}** - These are included in safe-to-outreach list.")

with st.expander("📋 View detailed breakdown by source"):
    permission_issues = []
    for src in active_sources:
        count = source_counts.get(src["label"], 0)
        safe = safe_domains.get(src["label"], 0)
        is_disavow = src.get("is_disavow", False)

        if is_disavow:
            # For disavow lists: all domains are excluded, show as "excluded" not "active"
            disavow_note = " 🚫 DISAVOW (always excluded)"
            st.write(f"  • **{src['label']}**: {count:,} total (all {count:,} excluded, 0 SAFE){disavow_note}")
        else:
            # Regular sources: show active vs SAFE
            active = count - safe
            status_msg = f"{count:,} total ({active:,} active, {safe:,} SAFE)"

            # Check if this source returned 0 and might have permission issues
            if count == 0:
                status_msg += " ⚠️"
                permission_issues.append(src["label"])

            st.write(f"  • **{src['label']}**: {status_msg}")

    # Show permission warning if any sources have issues
    if permission_issues:
        st.warning(f"⚠️ **Permission Issue Detected:** {', '.join(permission_issues)} returned 0 domains.")
        with st.expander("🔧 Troubleshooting Steps"):
            st.write("**If the base shows access in Airtable token settings but still returns 403:**")
            st.write("")
            st.write("**Most Common Causes:**")
            st.write("1. **Missing Scope:** Token needs `data.records:read` scope (not just `schema.bases:read`)")
            st.write("2. **Workspace Mismatch:** The database might be in a different workspace than the token has access to")
            st.write("3. **Token Mismatch:** Token in Streamlit secrets might be different from the one you're editing")
            st.write("")
            st.write("**Quick Fixes:**")
            st.write("1. In Airtable token settings, ensure `data.records:read` scope is enabled (check the box)")
            st.write("2. Verify the database is in the same workspace as other bases")
            st.write("3. Click 'Save changes' in Airtable token settings")
            st.write("4. Wait 10-30 seconds for changes to propagate")
            st.write("")

            # Test access for problematic bases
            for src in active_sources:
                if src["label"] in permission_issues:
                    st.write(f"**🔍 Testing {src['label']}:**")
                    with st.spinner(f"Testing access to {src['label']}..."):
                        success, msg = test_base_access(src["base_id"], src["table_id"])
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                            st.code(f"Base ID: {src['base_id']}\nTable ID: {src['table_id']}")
                            st.write("**Next Steps:**")
                            st.write("- Check if `data.records:read` scope is enabled in token settings")
                            st.write("- Verify the base is in the same workspace as other working bases")
                            st.write("- Try creating a new token with all scopes enabled")

new_to_outreach = sorted(d for d in new_domains if d not in existing)
reoutreach_in_list = len([d for d in new_to_outreach if d in reoutreach_candidates])
if reoutreach_in_list > 0:
    st.success(f"✅ {len(new_to_outreach)} new domains currently safe to outreach (pre-push check).")
    st.info(f"📊 **Breakdown:** {len(new_to_outreach) - reoutreach_in_list:,} completely new domains + {reoutreach_in_list:,} re-outreach candidates (3+ months old, only in {selected_vertical} Prospect-Data sources).")
else:
    st.success(f"✅ {len(new_to_outreach)} new domains currently safe to outreach (pre-push check).")

df_result = pd.DataFrame({"Domain": new_to_outreach})
st.dataframe(df_result, use_container_width=True)
st.download_button("⬇️ Download Prospects (CSV)", df_result.to_csv(index=False), "prospects.csv")

# ---------- Push to selected vertical's Prospect-Data base (duplicate-safe) ----------
if new_to_outreach:
    pushed_key = f"pushed_{selected_vertical}"
    if pushed_key not in st.session_state:
        st.session_state[pushed_key] = False
    disabled = st.session_state[pushed_key]

    st.write(f"Target for push → **{push_target_label}** (`{PUSH_BASE_ID}:{PUSH_TABLE_ID}`)")
    if st.button(f"📤 Push {len(new_to_outreach)} Prospects to Airtable (duplicate-safe)", disabled=disabled):
        with st.spinner("Re-checking latest records and creating new ones..."):
            latest_existing, _, _, latest_domain_to_sources, latest_domain_dates_by_source = fetch_existing_domains(
                active_sources,
                exclude_old_domains=True,
                months_threshold=12,
                return_source_mapping=True,
                prospect_data_labels=_prospect_labels_tuple
            )
            # Re-identify re-outreach candidates with latest data
            latest_reoutreach = set()
            if latest_domain_to_sources is not None and latest_domain_dates_by_source is not None and latest_domain_to_sources and latest_domain_dates_by_source:
                latest_reoutreach = identify_reoutreach_candidates(
                    set(new_to_outreach),
                    latest_domain_to_sources,
                    latest_domain_dates_by_source,
                    months_threshold=3,
                    prospect_data_labels=prospect_data_labels,
                    push_target_label=push_target_label
                )
                # Remove re-outreach candidates from latest_existing
                latest_existing = latest_existing - latest_reoutreach
            to_push = [d for d in new_to_outreach if d not in latest_existing]

            if len(to_push) < len(new_to_outreach):
                st.warning(f"⚠️ {len(new_to_outreach) - len(to_push)} domains were added by another process. Only {len(to_push)} will be pushed.")

            created = 0
            updated = 0
            skipped = 0
            errors  = 0
            error_details = []
            date_str = datetime.now().strftime("%Y-%m-%d")

            progress_bar = st.progress(0)
            total = len(to_push)

            for idx, d in enumerate(to_push):
                exists, record_id = domain_exists_in_prospect(d, push_table)

                # Check if this is a re-outreach candidate
                is_reoutreach = d in latest_reoutreach

                if exists:
                    if is_reoutreach and record_id:
                        # Update existing record for re-outreach candidates
                        try:
                            push_table.update(record_id, {
                                "Date": date_str,
                                "Added By Name": user_name,
                                "Added By Email": user_email
                            })
                            updated += 1
                            time.sleep(0.05)
                        except Exception as e:
                            errors += 1
                            error_details.append(f"{d} (update): {str(e)}")
                            logging.error(f"Error updating record for {d}: {e}")
                    else:
                        # Skip domains that exist but aren't re-outreach candidates
                        skipped += 1
                else:
                    # Create new record
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
            st.session_state[pushed_key] = True

            # Show results
            result_parts = [f"✅ **Created:** {created}"]
            if updated > 0:
                result_parts.append(f"🔄 **Updated (re-outreach):** {updated}")
            if skipped > 0:
                result_parts.append(f"⏭️ **Skipped (already existed):** {skipped}")
            if errors > 0:
                result_parts.append(f"⚠️ **Errors:** {errors}")
            st.success("  •  ".join(result_parts))

            if errors > 0 and error_details:
                with st.expander("⚠️ View error details"):
                    for err in error_details[:10]:  # Show first 10 errors
                        st.text(err)
                    if len(error_details) > 10:
                        st.text(f"... and {len(error_details) - 10} more errors")
