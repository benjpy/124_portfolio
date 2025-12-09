import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Portfolio Search", page_icon="sosv.png", layout="wide")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it in your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

EMBED_FILE = "embed.csv"
LOGO_FILE = "sosv.png"

import textwrap
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    /* Global Clean Look */
    .stApp {
        background-color: #f8f9fa; /* Very light subtle grey/white */
        font-family: 'Inter', sans-serif;
    }
    
    /* Search Input Styling */
    /* Force white background and black text to override Dark Mode defaults */
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        padding: 12px 16px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00A550 !important;
        box-shadow: 0 4px 10px rgba(0, 165, 80, 0.1) !important;
    }
    
    /* Result Card Styling */
    .result-card {
        background-color: white;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid #eaedf0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        border-color: #00A550;
    }
    
    /* Typography */
    .company-name {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 4px;
        text-decoration: none;
    }
    .company-name:hover {
        color: #00A550;
        text-decoration: underline;
    }
    .meta-tag {
        font-size: 0.85rem;
        color: #666;
        background-color: #f1f3f5;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;
        font-weight: 500;
    }
    .description-text {
        font-size: 1rem;
        color: #333;
        line-height: 1.6;
        margin-top: 12px;
    }
    .location-text {
        font-size: 0.9rem;
        color: #555;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 12px;
    }
    
    /* Header/Logo Area */
    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 30px;
        padding-top: 20px;
        text-align: center;
    }
    .logo-img {
        max-width: 150px;
        margin-bottom: 10px;
        object-fit: contain;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        border: none;
        background-color: #00A550;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover {
        background-color: #008741;
        box-shadow: 0 4px 12px rgba(0, 165, 80, 0.2);
    }
    /* Hide default Streamlit header and footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if not os.path.exists(EMBED_FILE):
        return None
    df = pd.read_csv(EMBED_FILE)
    # Convert string representation of list back to actual list if necessary
    # pandas read_csv might read lists as strings "[0.1, 0.2, ...]"
    # We need to evaluate them safely.
    import ast
    
    def parse_embedding(x):
        try:
            return ast.literal_eval(x)
        except:
            return None
            
    if "embedding" in df.columns:
        # Check if it's already a list (if parquet) or string (csv)
        if df["embedding"].notna().any() and isinstance(df["embedding"].iloc[0], str):
            df["embedding"] = df["embedding"].apply(parse_embedding)
            
    return df

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        return 0.0
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    if norma == 0 or normb == 0:
        return 0.0
    return dot / (norma * normb)

df = load_data()

# --- Header Section ---
if os.path.exists(LOGO_FILE):
    logo_b64 = get_base64_image(LOGO_FILE)
    st.markdown(f"""
        <div class="header-container">
            <img src="data:image/png;base64,{logo_b64}" class="logo-img" style="width: 150px; max-width: 150px;">
            <h1 style='color: #1e1e1e; margin: 0; padding: 0;'>Portfolio Search</h1>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("<h1 style='text-align: center; color: #1e1e1e;'>Portfolio Search</h1>", unsafe_allow_html=True)

if df is None:
    st.warning(f"Embedding file `{EMBED_FILE}` not found. Please run `generate_embeddings.py` first.")
    st.stop()

# --- Search Section ---
st.markdown("###") # Spacer
query = st.text_input("", placeholder="🔍 Search for companies (e.g., 'Biotech startups in NYC using AI')", label_visibility="collapsed")

if query:
    try:
        with st.spinner("Finding matches..."):
            query_embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=str(query),
                task_type="retrieval_query"
            )['embedding']
            
        # Filter Status
        filtered_df = df[df["Status"].isin(["Operating", "Written Down"])].copy()
        
        # Calculate similarities
        # Optimization: use matrix multiplication if possible, but loop is fine for <5k
        # Let's try to verify embeddings are valid lists
        valid_mask = filtered_df["embedding"].apply(lambda x: isinstance(x, list) and len(x) > 0)
        filtered_df = filtered_df[valid_mask]
        
        if filtered_df.empty:
            st.info("No matching operating companies found.")
        else:
            filtered_df["similarity"] = filtered_df["embedding"].apply(lambda x: cosine_similarity(query_embedding, x))
            
            # Sort
            results = filtered_df.sort_values(by="similarity", ascending=False)
            
            # Pagination
            ITEMS_PER_PAGE = 20
            
            if "page_number" not in st.session_state:
                st.session_state.page_number = 0
                
            # Reset page on new query
            # A simple way is to use a callback or just reset if query changed? 
            # Streamlit reruns on input change.
            
            # Let's do simple query parameter or session state for page
            # To detect query change, we can check session state
            if "last_query" not in st.session_state or st.session_state.last_query != query:
                st.session_state.page_number = 0
                st.session_state.last_query = query
            
            total_results = len(results)
            page = st.session_state.page_number
            start = page * ITEMS_PER_PAGE
            end = start + ITEMS_PER_PAGE            
            page_results = results.iloc[start:end]
            
            st.markdown(f"<div style='margin-bottom: 20px; color: #666; font-size: 0.9em;'>Found {total_results} companies. Showing {start+1}-{min(end, total_results)}.</div>", unsafe_allow_html=True)
            
            for _, row in page_results.iterrows():
                # Data Prep
                website = row['Website']
                if not str(website).startswith("http"):
                    website = f"https://{website}"
                    
                city = row.get('Primary Location (City)', '')
                country = row.get('Primary Location (Country)', '')
                location_str = f"{city}, {country}".strip(", ")
                
                desc = row.get('Description (SOSV)', row.get('Description', ''))
                
                # Render Card
                # We essentially minify the HTML to avoid any Markdown indentation issues.
                card_html = (
                    f'<div class="result-card">'
                    f'<div style="display: flex; justify-content: space-between; align-items: start;">'
                    f'<div><a href="{website}" target="_blank" class="company-name">{row["Name"]}</a>'
                    f'<div class="location-text">📍 {location_str}</div></div>'
                    f'<div style="text-align: right;"><span style="font-size: 0.8rem; color: #999; font-weight: 600;">Match: {int(row["similarity"]*100)}%</span></div>'
                    f'</div>'
                    f'<div style="margin-top: 8px; margin-bottom: 12px;">'
                    f'<span class="meta-tag">{row["Program Category"]}</span>'
                    f'<span class="meta-tag">{row["FA Code"]}</span>'
                    f'<span class="meta-tag" style="background-color: {"#e6f4ea" if row["Status"] == "Operating" else "#fce8e6"}; color: {"#137333" if row["Status"] == "Operating" else "#c5221f"};">{row["Status"]}</span>'
                    f'</div>'
                    f'<div class="description-text">{desc}</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
            
            # Pagination Controls
            col1, col2, col3 = st.columns([1, 8, 1])
            with col1:
                if page > 0:
                    if st.button("Previous"):
                        st.session_state.page_number -= 1
                        st.rerun()
            with col3:
                 if end < total_results:
                    if st.button("Next"):
                        st.session_state.page_number += 1
                        st.rerun()

    except Exception as e:
        st.error(f"Error during search: {e}")

else:
    # Empty State with a nice prompt or graphic if desired
    st.markdown("""
        <div style="text-align: center; margin-top: 50px; color: #888;">
            <p style="font-size: 1.2rem;">Enter keywords above to semantic search the portfolio.</p>
            <p style="font-size: 0.9rem;">Try searching for technologies, industries, or specific problems.</p>
        </div>
    """, unsafe_allow_html=True)
