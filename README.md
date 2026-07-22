# Prospect Filtering & Airtable Sync Tool

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
APP_PASSWORD   = "your_shared_access_password"
```

The app is password-protected: on load it prompts for `APP_PASSWORD` and grants
access only on a match. If `APP_PASSWORD` is not set, the tool fails closed and
refuses to run. Share the password only with authorized users.

3. Run locally:

```bash
streamlit run streamlit_airtable_prospect_tool.py
```

4. Deploy to Streamlit Cloud and add secrets in **App Settings → Secrets**.
