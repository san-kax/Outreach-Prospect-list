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

3. Configure Streamlit secrets (Streamlit Cloud → App Settings → Secrets):

```toml
airtable_token = "your_airtable_api_token"
```

4. Run the app locally:

```bash
streamlit run streamlit_airtable_prospect_tool.py
```

5. Deploy to **Streamlit Cloud**.

