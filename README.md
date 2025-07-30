# Prospect Filtering & Airtable Sync Tool (Table ID Version)

This Streamlit app allows you to upload a list of prospect domains, compare them against two Airtable databases (Backlink & Prospecting),
and identify domains that are **safe to outreach**.

### Features
- Uses **Airtable Table IDs** for reliable API access
- Upload CSV/Excel with `Domain` column
- Filters out existing Backlinks & Prospects
- Download CSV of new domains
- Push new domains to Airtable with:
  - `Date Added`
  - `Added By Name`
  - `Added By Email`

### Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure Streamlit secrets:

```toml
airtable_token = "your_airtable_pat_here"
```

3. Run locally:

```bash
streamlit run streamlit_airtable_prospect_tool_tableid.py
```

4. Deploy to Streamlit Cloud and add secrets in **App Settings → Secrets**.
