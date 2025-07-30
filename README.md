# Prospect Filtering & Airtable Sync Tool

This Streamlit app allows you to upload a list of prospect domains, compare them against two Airtable databases (Backlink & Prospecting), 
and identify domains that are **safe to outreach** (not present in either database).

## Features

- Upload CSV or Excel file with a `Domain` column.
- Automatically filters out domains already present in Backlink or Prospecting Airtable tables.
- Normalizes domains (lowercase, removes `www.`).
- Provides a downloadable CSV of new domains.
- **Two-step confirmation** before pushing new domains to Airtable.
- Automatically logs:
  - `Date Added`
  - `Added By Name`
  - `Added By Email`
- Prevents duplicate pushes in the same session.

## Setup Instructions

1. Clone this repository or upload the script to GitHub.
2. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.streamlit/secrets.toml` file with your Airtable API token:

```toml
airtable_token = "your_airtable_api_token"
```

4. Update the script with your Airtable Base IDs and Table Names:

```python
BACKLINK_BASE_ID = "your_backlink_base_id"
BACKLINK_TABLE_NAME = "your_backlink_table_name"
PROSPECT_BASE_ID = "your_prospect_base_id"
PROSPECT_TABLE_NAME = "your_prospect_table_name"
```

5. Run the app locally:

```bash
streamlit run streamlit_airtable_prospect_tool.py
```

6. Deploy to **Streamlit Cloud** or any hosting provider.

## Usage

1. Enter your **Name** and **Email**.
2. Upload a CSV or Excel file with a `Domain` column.
3. Review the filtered results.
4. Optionally download the CSV of new domains.
5. Click **Push to Airtable** → **Confirm Push** to upload to your Prospecting Airtable.

## Output in Airtable

Each pushed domain will have:

- Domain
- Date Added
- Added By Name
- Added By Email
