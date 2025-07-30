import streamlit as st
import pandas as pd
from pyairtable import Table
from datetime import datetime

# Airtable configuration (keep secrets in .streamlit/secrets.toml)
AIRTABLE_TOKEN = st.secrets["airtable_token"]

BACKLINK_BASE_ID = "appZEmIU..."       # Replace with actual
BACKLINK_TABLE_NAME = "Imported table"
PROSPECT_BASE_ID = "appHdhjsW..."      # Replace with actual
PROSPECT_TABLE_NAME = "Imported table"

# Connect to Airtable tables
backlink_table = Table(AIRTABLE_TOKEN, BACKLINK_BASE_ID, BACKLINK_TABLE_NAME)
prospect_table = Table(AIRTABLE_TOKEN, PROSPECT_BASE_ID, PROSPECT_TABLE_NAME)

st.title("🔗 Prospect Filtering & Airtable Sync Tool")

# --- User Authentication ---
st.subheader("👤 User Info")
user_name = st.text_input("Enter your name:")
user_email = st.text_input("Enter your email:")

if not user_name or not user_email:
    st.warning("⚠️ Please provide your name and email to continue.")
    st.stop()

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your prospect domains (CSV/Excel)", type=["csv", "xlsx"])

# Track session state
if "pushed" not in st.session_state:
    st.session_state.pushed = False
if "confirm_push" not in st.session_state:
    st.session_state.confirm_push = False

if uploaded_file:
    # Read file
    df_new = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    # Check for 'Domain' column
    if 'Domain' not in df_new.columns:
        st.error("Your file must have a 'Domain' column!")
    else:
        # Normalize domains (remove www, lowercase)
        new_domains = (
            df_new['Domain']
            .str.strip()
            .str.lower()
            .str.replace(r"^www\.", "", regex=True)
            .dropna()
            .unique()
        )

        st.info("Fetching existing domains from Airtable...")

        # Fetch Airtable records
        backlinks = backlink_table.all()
        prospects = prospect_table.all()

        backlink_domains = {
            r['fields'].get('Domain', '').strip().lower().replace("www.", "")
            for r in backlinks
        }
        prospect_domains = {
            r['fields'].get('Domain', '').strip().lower().replace("www.", "")
            for r in prospects
        }

        # Combine and filter
        all_existing = backlink_domains.union(prospect_domains)
        new_to_outreach = [d for d in new_domains if d not in all_existing]

        # Show results
        st.success(f"✅ {len(new_to_outreach)} new domains found that are safe to outreach.")
        df_result = pd.DataFrame({'Domain': new_to_outreach})
        st.dataframe(df_result)

        # Download CSV
        st.download_button("⬇️ Download CSV", df_result.to_csv(index=False), "outreach_candidates.csv")

        # --- Push to Airtable ---
        if len(new_to_outreach) > 0 and not st.session_state.pushed:
            
            # Step 1: Ask for confirmation
            if not st.session_state.confirm_push:
                if st.button("📤 Push New Domains to Airtable"):
                    st.session_state.confirm_push = True
                    st.warning("⚠️ Please confirm: Are you sure you want to push these domains to Airtable?")
            
            # Step 2: Show final confirmation button
            elif st.session_state.confirm_push:
                if st.button("✅ Confirm Push to Airtable"):
                    added_count = 0
                    today = datetime.now().strftime("%Y-%m-%d")

                    for domain in new_to_outreach:
                        try:
                            prospect_table.create({
                                "Domain": domain,
                                "Date Added": today,
                                "Added By Name": user_name,
                                "Added By Email": user_email
                            })
                            added_count += 1
                        except Exception as e:
                            st.error(f"❌ Failed to add {domain}: {e}")

                    st.success(f"🎉 Successfully added {added_count} new domains to Prospecting Airtable!")
                    st.session_state.pushed = True

        elif st.session_state.pushed:
            st.info("✅ Domains have already been pushed to Airtable in this session.")
