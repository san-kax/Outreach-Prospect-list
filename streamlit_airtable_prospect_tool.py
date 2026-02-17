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
    st.error("Missing Airtable token. Add AIRTABLE_TOKEN (or airtable_token) in Streamlit secrets or env.")
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
            return True, "Access OK - Can read schema and data"
        except Exception as data_error:
            error_str = str(data_error)
            if "403" in error_str or "Forbidden" in error_str:
                if schema_ok:
                    return False, f"403 Forbidden - Can see base schema but cannot read records. Token may need 'data.records:read' scope."
                else:
                    return False, f"403 Forbidden - Cannot access base. Check: 1) Token scopes 2) Workspace permissions 3) Base is in correct workspace"
            return False, f"Error: {error_str[:150]}"
    except Exception as e:
        error_str = str(e)
        if "403" in error_str or "Forbidden" in error_str:
            return False, f"403 Forbidden - Token may not have access to base {base_id}"
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
    "States": {
        "prospect_base_id": "appzVpYiLO90EgRgj",
        "prospect_table_id": "tbliCOQZY9RICLsLP",
        "prospect_label": "Prospect-Data-States-Sites",
    },
}

# ---- Database / Live Link sources (domain here = live link confirmed) ----
# Each mapped to the vertical(s) it belongs to for per-vertical threshold logic.
# "verticals" list indicates which vertical(s) this database represents.
DATABASE_SOURCES = [
    {"label": "GDC-Database",        "base_id": "appUoOvkqzJvyyMvC", "table_id": "tbliCOQZY9RICLsLP", "is_disavow": False, "is_database": True, "verticals": ["GDC"]},
    {"label": "WB-Database",         "base_id": "appueIgn44RaVH6ot", "table_id": "tbl3vMYv4RzKfuBf4", "is_disavow": False, "is_database": True, "verticals": ["WhichBingo"]},
    {"label": "Freebets-Database",   "base_id": "appFBasaCUkEKtvpV", "table_id": "tblmTREzfIswOuA0F", "is_disavow": False, "is_database": True, "verticals": ["Freebets"]},
    {"label": "BonusFinder-Database", "base_id": "appZEyAoVubSrBl9w", "table_id": "tbl4pzZFkzfKLhtkK", "is_disavow": False, "is_database": True, "verticals": ["BonusFinder"]},
    {"label": "Casinos-Links",       "base_id": "appay75NrffUxBMbM", "table_id": "tblx8ZGluvQ9cWdXh", "is_disavow": False, "is_database": True, "verticals": ["Casinos"]},
    {"label": "Local States Vertical Live Links", "base_id": "app08yUTcPhJVPxCI", "table_id": "tbllmyX2xNVXMEEnc", "is_disavow": False, "is_database": True, "verticals": ["States"]},
    {"label": "Sports Vertical Bookies.com and Rotowire", "base_id": "appDFsy6RWw5TRNH6", "table_id": "tbl8whN06WyCOo5uk", "is_disavow": False, "is_database": True, "verticals": ["Bookies", "Rotowire"]},
]

# ---- Disavow / Rejected sources (always blocked, never reusable) ----
DISAVOW_SOURCES = [
    {"label": "GDC-Disavow-List",    "base_id": "appJTJQwjHRaAyLkw", "table_id": "tbliCOQZY9RICLsLP", "is_disavow": True, "is_database": False},
    {"label": "GDC-Disavow-List-1",  "base_id": "appEEpV8mgLcBMQLE", "table_id": "tbliCOQZY9RICLsLP", "is_disavow": True, "is_database": False},
    {"label": "Outreach-Rejected-Sites", "base_id": "appTf6MmZDgouu8SN", "table_id": "tbliCOQZY9RICLsLP", "is_disavow": True, "is_database": False},
]

# ---- Build a set of ALL prospect-data labels across all verticals ----
ALL_PROSPECT_LABELS: set[str] = set()
for _vconfig in VERTICALS.values():
    ALL_PROSPECT_LABELS.add(_vconfig["prospect_label"])
    if "extra_prospect" in _vconfig:
        ALL_PROSPECT_LABELS.add(_vconfig["extra_prospect"]["label"])

# ---- Build a set of ALL database labels ----
ALL_DATABASE_LABELS: set[str] = {src["label"] for src in DATABASE_SOURCES}

# ---- Build a set of ALL disavow labels ----
ALL_DISAVOW_LABELS: set[str] = {src["label"] for src in DISAVOW_SOURCES}

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
    date_field_names = [
        "Date", "date", "Added Date", "Added date", "Publication Date",
        "Publication date", "Created Date", "Created date", "First seen",
        "Last seen", "Date Added", "Date added"
    ]
    for field_name in date_field_names:
        if field_name in record_fields:
            return record_fields[field_name]
    return None

def make_tz_naive(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-naive for safe comparison."""
    if dt is not None and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt

def get_domain_age_months(record_fields: dict) -> float | None:
    """Get domain age in months from its date field. Returns None if no date found."""
    date_field = find_date_field(record_fields)
    if not date_field:
        return None
    parsed_date = parse_date(str(date_field))
    if not parsed_date:
        return None
    parsed_date = make_tz_naive(parsed_date)
    now = datetime.now()
    delta = now - parsed_date
    return delta.days / 30.0

def is_domain_safe_to_reuse(record_fields: dict, months_threshold: int = 12) -> bool:
    """Check if domain is older than threshold months and can be reused."""
    age_months = get_domain_age_months(record_fields)
    if age_months is None:
        # If no date, consider it as old (safe to reuse) - conservative approach
        return True
    return age_months >= months_threshold


def fetch_single_source(
    src: dict,
    exclude_old_domains: bool,
    months_threshold: int,
    return_domain_dates: bool = False,
    date_tracking_labels: set[str] | None = None,
) -> tuple[str, set[str], int, int, str | None, dict[str, datetime] | None]:
    """Fetch domains from a single Airtable source.

    Returns (label, domains_set, count, safe_count, field_used, domain_dates_dict).

    Args:
        return_domain_dates: If True, also return a dict mapping domain -> date for tracked sources
        date_tracking_labels: Set of labels for which to track domain dates (Prospect-Data + Database sources)
    """
    try:
        table = api.base(src["base_id"]).table(src["table_id"])
        is_disavow_list = src.get("is_disavow", False)
        _date_labels = date_tracking_labels or set()
        should_track_dates = src["label"] in _date_labels

        # Try to fetch only needed fields for better performance
        possible_domain_fields = ["Domain", "domain", "A Domain", "Live Link", "Referring page URL"]
        possible_date_fields = ["Date", "date", "Added Date", "Publication Date", "Created Date", "First seen", "Last seen"]

        # First, try fetching with specific fields (faster)
        try:
            all_fields_to_fetch = list(set(possible_domain_fields + possible_date_fields))
            records = table.all(fields=all_fields_to_fetch)
        except Exception as e1:
            error_str = str(e1)
            if "403" in error_str or "Forbidden" in error_str or "INVALID_PERMISSIONS" in error_str:
                logging.error(f"Permission denied for {src['label']} (base: {src['base_id']}, table: {src['table_id']})")
                raise
            # If that fails for other reasons, fetch all fields (slower but more reliable)
            try:
                records = table.all()
            except Exception as e2:
                error_str2 = str(e2)
                if "403" in error_str2 or "Forbidden" in error_str2 or "INVALID_PERMISSIONS" in error_str2:
                    logging.error(f"Permission denied for {src['label']} (base: {src['base_id']}, table: {src['table_id']})")
                raise

        domains_set: set[str] = set()
        domain_dates: dict[str, datetime] = {}
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

                # Store date for tracked sources if requested
                if return_domain_dates and should_track_dates:
                    date_field = find_date_field(fields)
                    if date_field:
                        parsed_date = parse_date(str(date_field))
                        if parsed_date:
                            parsed_date = make_tz_naive(parsed_date)
                            domain_dates[d] = parsed_date

                # Disavow lists: ALWAYS exclude, no age exception
                if is_disavow_list:
                    domains_set.add(d)
                else:
                    # Regular sources: apply age-based threshold
                    if exclude_old_domains and is_domain_safe_to_reuse(fields, months_threshold):
                        safe_count += 1
                    else:
                        domains_set.add(d)

        if count == 0 and fields_logged:
            logging.warning(f"{src['label']} returned 0 domains - available fields: {list(fields.keys())}")

        return (src["label"], domains_set, count, safe_count, domain_field_found, domain_dates if return_domain_dates and should_track_dates else None)

    except Exception as e:
        error_str = str(e)
        if "403" in error_str or "Forbidden" in error_str or "INVALID_PERMISSIONS" in error_str:
            logging.error(f"Permission denied for {src['label']} (base: {src['base_id']}, table: {src['table_id']})")
        else:
            logging.error(f"Error fetching from {src['label']} ({src['base_id']}): {e}")
            logging.error(traceback.format_exc())
        return (src["label"], set(), 0, 0, None, None)


@st.cache_data(ttl=300)
def fetch_existing_domains(
    selected_sources: list[dict],
    show_progress: bool = False,
    exclude_old_domains: bool = True,
    months_threshold: int = 12,
    return_source_mapping: bool = False,
    date_tracking_labels: tuple[str, ...] = (),
) -> tuple[set[str], dict[str, int], dict[str, int], dict[str, set[str]] | None, dict[str, dict[str, datetime]] | None]:
    """Read 'Domain' from all selected bases/tables and return a unified normalized set.

    Uses parallel processing to fetch from multiple sources simultaneously for better performance.

    Args:
        selected_sources: List of source dictionaries with base_id, table_id, label
        show_progress: Whether to log progress
        exclude_old_domains: If True, exclude domains older than months_threshold (SAFE to reuse)
        months_threshold: Number of months after which a domain is considered SAFE to reuse
        return_source_mapping: If True, also return domain_to_sources mapping and domain_dates_by_source
        date_tracking_labels: Tuple of labels for which to track domain dates

    Returns:
        tuple: (all_domains_set, source_counts_dict, safe_domains_dict, domain_to_sources, domain_dates_by_source)
    """
    all_domains: set[str] = set()
    source_counts: dict[str, int] = {}
    safe_domains: dict[str, int] = {}
    domain_to_sources: dict[str, set[str]] = {} if return_source_mapping else None
    domain_dates_by_source: dict[str, dict[str, datetime]] = {} if return_source_mapping else None
    _date_labels_set = set(date_tracking_labels)

    # Process sources in parallel for better performance
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_source = {
            executor.submit(fetch_single_source, src, exclude_old_domains, months_threshold, return_source_mapping, _date_labels_set): src
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

                # Store domain dates for tracked sources
                if domain_dates:
                    domain_dates_by_source[label] = domain_dates

            if show_progress:
                active = count - safe_count
                field_info = f" (using field: {field_used})" if field_used else ""
                logging.info(f"Fetched {count:,} domains from {label} - {active:,} active, {safe_count:,} SAFE{field_info}")

    return all_domains, source_counts, safe_domains, domain_to_sources, domain_dates_by_source


def apply_smart_dedup_rules(
    all_domains: set[str],
    domain_to_sources: dict[str, set[str]],
    domain_dates_by_source: dict[str, dict[str, datetime]],
    selected_vertical: str,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Apply the 4 business rules to determine which domains to block/allow.

    Rules:
        1. No simultaneous outreach - domains in ANY Prospect-Data source are blocked (handled by mandatory sources)
        2. Live link (Database) sites safe after 12 months for SAME vertical
        3. Prospect-only domains (no live link) safe after 3 months for ALL builders
        4. Live link (Database) sites safe after 4 months for DIFFERENT vertical

    Args:
        all_domains: Set of all domains currently in the blocking set (from fetch)
        domain_to_sources: Dict mapping domain -> set of source labels
        domain_dates_by_source: Dict mapping source label -> dict(domain -> date)
        selected_vertical: The vertical the current builder selected

    Returns:
        tuple: (
            final_blocked: domains that should remain blocked,
            safe_prospect_3m: domains safe due to 3-month no-result rule (Rule 3),
            safe_db_diff_vertical_4m: domains safe due to 4-month different-client rule (Rule 4),
            safe_db_same_vertical_12m: domains safe due to 12-month same-vertical rule (Rule 2),
        )
    """
    now = datetime.now()
    threshold_3m = now - timedelta(days=3 * 30)
    threshold_4m = now - timedelta(days=4 * 30)
    threshold_12m = now - timedelta(days=12 * 30)

    # Build lookup: which database labels belong to which verticals
    db_label_to_verticals: dict[str, list[str]] = {}
    for db_src in DATABASE_SOURCES:
        db_label_to_verticals[db_src["label"]] = db_src["verticals"]

    safe_prospect_3m: set[str] = set()       # Rule 3: no-result after 3 months
    safe_db_diff_vertical_4m: set[str] = set()  # Rule 4: live link, different client, 4+ months
    safe_db_same_vertical_12m: set[str] = set()  # Rule 2: live link, same client, 12+ months

    for domain in all_domains:
        sources = domain_to_sources.get(domain, set())
        if not sources:
            continue

        # Classify which source types this domain appears in
        in_prospect_sources = sources.intersection(ALL_PROSPECT_LABELS)
        in_database_sources = sources.intersection(ALL_DATABASE_LABELS)
        in_disavow_sources = sources.intersection(ALL_DISAVOW_LABELS)

        # Disavow: ALWAYS blocked, skip any other logic
        if in_disavow_sources:
            continue

        # --- Rule 3: Domain is ONLY in Prospect-Data (no live link), older than 3 months ---
        if in_prospect_sources and not in_database_sources:
            # Check if ALL prospect entries for this domain are older than 3 months
            all_old_enough = True
            has_date = False
            for plabel in in_prospect_sources:
                if plabel in domain_dates_by_source:
                    dates = domain_dates_by_source[plabel]
                    if domain in dates:
                        has_date = True
                        domain_date = make_tz_naive(dates[domain])
                        if domain_date >= threshold_3m:
                            all_old_enough = False
                            break

            if has_date and all_old_enough:
                safe_prospect_3m.add(domain)
            continue  # Don't apply database rules to prospect-only domains

        # --- Rules 2 & 4: Domain is in Database source(s) (live link confirmed) ---
        if in_database_sources:
            # Determine if the database is same-vertical or different-vertical
            is_same_vertical_db = False
            is_diff_vertical_db = False

            for db_label in in_database_sources:
                db_verticals = db_label_to_verticals.get(db_label, [])
                if selected_vertical in db_verticals:
                    is_same_vertical_db = True
                else:
                    is_diff_vertical_db = True

            # Get the most recent date across all database entries for this domain
            most_recent_db_date = None
            for db_label in in_database_sources:
                if db_label in domain_dates_by_source:
                    dates = domain_dates_by_source[db_label]
                    if domain in dates:
                        d_date = make_tz_naive(dates[domain])
                        if most_recent_db_date is None or d_date > most_recent_db_date:
                            most_recent_db_date = d_date

            if most_recent_db_date is None:
                # No date found - treat as old/safe (conservative: allow reuse)
                if is_same_vertical_db:
                    safe_db_same_vertical_12m.add(domain)
                elif is_diff_vertical_db:
                    safe_db_diff_vertical_4m.add(domain)
                continue

            # Rule 2: Same vertical, 12+ months old -> safe
            if is_same_vertical_db and not is_diff_vertical_db:
                if most_recent_db_date < threshold_12m:
                    safe_db_same_vertical_12m.add(domain)

            # Rule 4: Different vertical only, 4+ months old -> safe
            elif is_diff_vertical_db and not is_same_vertical_db:
                if most_recent_db_date < threshold_4m:
                    safe_db_diff_vertical_4m.add(domain)

            # Both same AND different vertical databases have this domain
            elif is_same_vertical_db and is_diff_vertical_db:
                # Must satisfy the stricter rule (12 months for same vertical)
                if most_recent_db_date < threshold_12m:
                    safe_db_same_vertical_12m.add(domain)

    # Combine all safe domains to remove from blocked set
    all_safe = safe_prospect_3m | safe_db_diff_vertical_4m | safe_db_same_vertical_12m
    final_blocked = all_domains - all_safe

    return final_blocked, safe_prospect_3m, safe_db_diff_vertical_4m, safe_db_same_vertical_12m


def _escape_for_formula(val: str) -> str:
    """Escape special characters for Airtable formula strings."""
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
        return True, None

# ---------------- UI ----------------
st.title("Prospect Filtering & Airtable Sync")

st.subheader("User")
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
    st.warning("Please provide your name and email to continue.")
    st.stop()

if not is_valid_email(user_email):
    st.error("Please provide a valid email address.")
    st.stop()

# ---------- Vertical selection ----------
st.subheader("Vertical")
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

# Build the prospect source (always checked - the push target itself)
prospect_source = {
    "label": vertical_config["prospect_label"],
    "base_id": vertical_config["prospect_base_id"],
    "table_id": vertical_config["prospect_table_id"],
    "is_disavow": False,
    "is_database": False,
}

# Build the set of "prospect data labels" for this vertical (used for re-outreach identification)
current_vertical_prospect_labels = {vertical_config["prospect_label"]}

# Add extra prospect source if it exists (e.g., GDC has Prospect-Data-GDC as secondary)
extra_prospect_sources = []
if "extra_prospect" in vertical_config:
    extra = dict(vertical_config["extra_prospect"])
    extra["is_disavow"] = False
    extra["is_database"] = False
    extra_prospect_sources = [extra]
    current_vertical_prospect_labels.add(extra["label"])

# ---- Build MANDATORY sources (cannot be deselected) ----
# All other verticals' Prospect-Data bases are mandatory for Rule 1 (no simultaneous outreach)
mandatory_prospect_sources = []
for vname, vconfig in VERTICALS.items():
    if vname != selected_vertical:
        mandatory_prospect_sources.append({
            "label": vconfig["prospect_label"],
            "base_id": vconfig["prospect_base_id"],
            "table_id": vconfig["prospect_table_id"],
            "is_disavow": False,
            "is_database": False,
        })
        # Include extra prospect sources from other verticals too
        if "extra_prospect" in vconfig:
            ep = dict(vconfig["extra_prospect"])
            ep["is_disavow"] = False
            ep["is_database"] = False
            mandatory_prospect_sources.append(ep)

# All Database, Disavow sources are also mandatory
mandatory_db_sources = [dict(src) for src in DATABASE_SOURCES]
mandatory_disavow_sources = [dict(src) for src in DISAVOW_SOURCES]

# Combine all mandatory sources (these cannot be unchecked)
mandatory_sources = (
    [prospect_source]
    + extra_prospect_sources
    + mandatory_prospect_sources
    + mandatory_db_sources
    + mandatory_disavow_sources
)

st.caption(
    f"`{push_target_label}` is the push target. "
    f"All Prospect-Data bases, Database/Live Link sources, and Disavow lists are always checked. Push goes to `{push_target_label}` only."
)

# ---------- Source display (all mandatory, no multiselect) ----------
st.write("**Deduplication Sources (all mandatory):**")

# Group sources for display
with st.expander("View all deduplication sources"):
    st.markdown(f"**Push Target:** `{prospect_source['label']}`")
    if extra_prospect_sources:
        st.markdown(f"**Extra Prospect Source:** `{extra_prospect_sources[0]['label']}`")

    st.markdown("---")
    st.markdown("**All Prospect-Data Sources (Rule 1 - no simultaneous outreach):**")
    for src in mandatory_prospect_sources:
        st.markdown(f"- `{src['label']}`")

    st.markdown("---")
    st.markdown("**Database / Live Link Sources (Rules 2 & 4):**")
    for src in mandatory_db_sources:
        verticals_str = ", ".join(src.get("verticals", []))
        same_or_diff = "SAME vertical" if selected_vertical in src.get("verticals", []) else "different vertical"
        st.markdown(f"- `{src['label']}` ({verticals_str}) - *{same_or_diff}*")

    st.markdown("---")
    st.markdown("**Disavow / Rejected Lists (always blocked):**")
    for src in mandatory_disavow_sources:
        st.markdown(f"- `{src['label']}`")

active_sources = mandatory_sources

st.info(f"**Active sources: {len(active_sources)}** - All sources are mandatory to enforce outreach rules.")

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

st.write(f"Uploaded rows: **{len(raw)}** | After normalization: **{len(new_domains)}**")

# ---------- Display rules ----------
st.subheader("Active Rules")
st.markdown("""
| Rule | Description | Threshold |
|------|-------------|-----------|
| **Rule 1** | No builders outreach to same site simultaneously | All Prospect-Data sources checked (mandatory) |
| **Rule 2** | Live link sites can be reused by same vertical | **12 months** after link confirmed |
| **Rule 3** | No-result prospects safe for all builders | **3 months** after outreach with no live link |
| **Rule 4** | Live link sites can be reused by different vertical | **4 months** after link confirmed |
| **Disavow** | Rejected/disavowed sites never reusable | Always blocked |
""")

# ---------- Build date tracking labels (all prospect + all database sources) ----------
date_tracking_labels = tuple(sorted(ALL_PROSPECT_LABELS | ALL_DATABASE_LABELS))

# ---------- Fetch from all sources ----------
with st.spinner("Fetching existing domains from Airtable..."):
    try:
        # Use a large threshold (120 months) to fetch ALL domains with their dates.
        # The smart dedup rules will apply the correct per-source thresholds afterwards.
        existing, source_counts, safe_domains_raw, domain_to_sources, domain_dates_by_source = fetch_existing_domains(
            active_sources,
            show_progress=True,
            exclude_old_domains=False,  # Don't apply blanket age filtering - we do it ourselves
            months_threshold=120,
            return_source_mapping=True,
            date_tracking_labels=date_tracking_labels,
        )
    except Exception as e:
        st.error(f"Error fetching domains: {e}")
        logging.error(f"Error in fetch_existing_domains: {e}")
        logging.error(traceback.format_exc())
        existing = set()
        source_counts = {src["label"]: 0 for src in active_sources}
        safe_domains_raw = {src["label"]: 0 for src in active_sources}
        domain_to_sources = {}
        domain_dates_by_source = {}

# ---------- Apply smart dedup rules ----------
if domain_to_sources and domain_dates_by_source is not None:
    blocked_domains, safe_prospect_3m, safe_db_diff_4m, safe_db_same_12m = apply_smart_dedup_rules(
        existing,
        domain_to_sources,
        domain_dates_by_source,
        selected_vertical,
    )
else:
    blocked_domains = existing
    safe_prospect_3m = set()
    safe_db_diff_4m = set()
    safe_db_same_12m = set()

# ---------- Calculate totals ----------
total_domains = sum(source_counts.values())
total_blocked = len(blocked_domains)
total_safe_3m = len(safe_prospect_3m)
total_safe_4m = len(safe_db_diff_4m)
total_safe_12m = len(safe_db_same_12m)

# ---------- Display results ----------
st.subheader("Results")
st.info(f"**Total domains across all sources:** {total_domains:,}")
st.error(f"**Blocked (active duplicates):** {total_blocked:,}")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rule 2: Same vertical 12m+", f"{total_safe_12m:,}", help="Live link sites older than 12 months, safe for same vertical")
with col2:
    st.metric("Rule 3: No-result 3m+", f"{total_safe_3m:,}", help="Prospect-only domains older than 3 months with no live link")
with col3:
    st.metric("Rule 4: Diff vertical 4m+", f"{total_safe_4m:,}", help="Live link sites older than 4 months, safe for different vertical")

# Detailed breakdown
with st.expander("View detailed breakdown by source"):
    permission_issues = []

    st.markdown("**Prospect-Data Sources:**")
    for src in active_sources:
        if src["label"] in ALL_PROSPECT_LABELS:
            count = source_counts.get(src["label"], 0)
            if count == 0:
                permission_issues.append(src["label"])
                st.write(f"  - **{src['label']}**: {count:,} domains (warning: 0 domains)")
            else:
                st.write(f"  - **{src['label']}**: {count:,} domains")

    st.markdown("---")
    st.markdown("**Database / Live Link Sources:**")
    for src in active_sources:
        if src.get("is_database"):
            count = source_counts.get(src["label"], 0)
            verticals_str = ", ".join(src.get("verticals", []))
            same_or_diff = "SAME" if selected_vertical in src.get("verticals", []) else "DIFF"
            threshold = "12 months" if same_or_diff == "SAME" else "4 months"
            if count == 0:
                permission_issues.append(src["label"])
                st.write(f"  - **{src['label']}** [{same_or_diff}, {threshold}]: {count:,} domains (warning: 0 domains)")
            else:
                st.write(f"  - **{src['label']}** [{same_or_diff}, {threshold}]: {count:,} domains")

    st.markdown("---")
    st.markdown("**Disavow / Rejected Sources:**")
    for src in active_sources:
        if src.get("is_disavow"):
            count = source_counts.get(src["label"], 0)
            st.write(f"  - **{src['label']}**: {count:,} domains (always blocked)")

    # Show permission warning if any sources have issues
    if permission_issues:
        st.warning(f"**Permission Issue Detected:** {', '.join(permission_issues)} returned 0 domains.")
        with st.expander("Troubleshooting Steps"):
            st.write("**Most Common Causes:**")
            st.write("1. **Missing Scope:** Token needs `data.records:read` scope")
            st.write("2. **Workspace Mismatch:** Database might be in a different workspace")
            st.write("3. **Token Mismatch:** Token in Streamlit secrets might differ from configured one")
            st.write("")
            st.write("**Quick Fixes:**")
            st.write("1. In Airtable token settings, ensure `data.records:read` scope is enabled")
            st.write("2. Verify the database is in the same workspace as other bases")
            st.write("3. Click 'Save changes' in Airtable token settings")
            st.write("4. Wait 10-30 seconds for changes to propagate")

            for src in active_sources:
                if src["label"] in permission_issues:
                    st.write(f"**Testing {src['label']}:**")
                    with st.spinner(f"Testing access to {src['label']}..."):
                        success, msg = test_base_access(src["base_id"], src["table_id"])
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                            st.code(f"Base ID: {src['base_id']}\nTable ID: {src['table_id']}")

# ---------- Filter uploaded domains ----------
new_to_outreach = sorted(d for d in new_domains if d not in blocked_domains)

# Categorize safe domains that are in the upload
reoutreach_3m_in_list = [d for d in new_to_outreach if d in safe_prospect_3m]
reoutreach_4m_in_list = [d for d in new_to_outreach if d in safe_db_diff_4m]
reoutreach_12m_in_list = [d for d in new_to_outreach if d in safe_db_same_12m]
completely_new = [d for d in new_to_outreach if d not in safe_prospect_3m and d not in safe_db_diff_4m and d not in safe_db_same_12m]

st.success(f"**{len(new_to_outreach)}** domains safe to outreach (pre-push check).")

if reoutreach_3m_in_list or reoutreach_4m_in_list or reoutreach_12m_in_list:
    st.markdown("**Breakdown:**")
    st.write(f"- **{len(completely_new):,}** completely new domains")
    if reoutreach_3m_in_list:
        st.write(f"- **{len(reoutreach_3m_in_list):,}** re-outreach candidates (Rule 3: no live link after 3+ months)")
    if reoutreach_4m_in_list:
        st.write(f"- **{len(reoutreach_4m_in_list):,}** available from different vertical (Rule 4: live link 4+ months ago)")
    if reoutreach_12m_in_list:
        st.write(f"- **{len(reoutreach_12m_in_list):,}** available from same vertical (Rule 2: live link 12+ months ago)")

df_result = pd.DataFrame({"Domain": new_to_outreach})
st.dataframe(df_result, use_container_width=True)
st.download_button("Download Prospects (CSV)", df_result.to_csv(index=False), "prospects.csv")

# ---------- Push to selected vertical's Prospect-Data base (duplicate-safe) ----------
if new_to_outreach:
    pushed_key = f"pushed_{selected_vertical}"
    if pushed_key not in st.session_state:
        st.session_state[pushed_key] = False
    disabled = st.session_state[pushed_key]

    st.write(f"Target for push: **{push_target_label}** (`{PUSH_BASE_ID}:{PUSH_TABLE_ID}`)")
    if st.button(f"Push {len(new_to_outreach)} Prospects to Airtable (duplicate-safe)", disabled=disabled):
        with st.spinner("Re-checking latest records and creating new ones..."):
            # CLEAR CACHE before re-check to get fresh data (fixes stale cache bug)
            fetch_existing_domains.clear()

            latest_existing, _, _, latest_domain_to_sources, latest_domain_dates_by_source = fetch_existing_domains(
                active_sources,
                exclude_old_domains=False,
                months_threshold=120,
                return_source_mapping=True,
                date_tracking_labels=date_tracking_labels,
            )

            # Re-apply smart rules with fresh data
            if latest_domain_to_sources and latest_domain_dates_by_source is not None:
                latest_blocked, _, _, _ = apply_smart_dedup_rules(
                    latest_existing,
                    latest_domain_to_sources,
                    latest_domain_dates_by_source,
                    selected_vertical,
                )
            else:
                latest_blocked = latest_existing

            to_push = [d for d in new_to_outreach if d not in latest_blocked]

            if len(to_push) < len(new_to_outreach):
                st.warning(f"{len(new_to_outreach) - len(to_push)} domains were added by another process. Only {len(to_push)} will be pushed.")

            # Identify which domains in to_push are re-outreach (already exist in push target)
            # vs completely new (need to be created)
            reoutreach_set = safe_prospect_3m | safe_db_diff_4m | safe_db_same_12m

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

                is_reoutreach = d in reoutreach_set

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

                if total > 0:
                    progress_bar.progress((idx + 1) / total)

            progress_bar.empty()
            st.session_state[pushed_key] = True

            # Show results
            result_parts = [f"**Created:** {created}"]
            if updated > 0:
                result_parts.append(f"**Updated (re-outreach):** {updated}")
            if skipped > 0:
                result_parts.append(f"**Skipped (already existed):** {skipped}")
            if errors > 0:
                result_parts.append(f"**Errors:** {errors}")
            st.success("  |  ".join(result_parts))

            if errors > 0 and error_details:
                with st.expander("View error details"):
                    for err in error_details[:10]:
                        st.text(err)
                    if len(error_details) > 10:
                        st.text(f"... and {len(error_details) - 10} more errors")
