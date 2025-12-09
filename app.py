import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Portfolio Search", layout="wide")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it in your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

EMBED_FILE = "embed.csv"

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
        if isinstance(df["embedding"].iloc[0], str):
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

st.title("Portfolio Semantic Search")

if df is None:
    st.warning(f"Embedding file `{EMBED_FILE}` not found. Please run `generate_embeddings.py` first.")
    st.stop()

query = st.text_input("Enter your search query:", placeholder="e.g., AI companies in bio-tech")

if query:
    try:
        with st.spinner("Generating embedding for query..."):
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
            st.warning("No operating companies found with valid embeddings.")
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
            
            # Display
            total_results = len(results)
            start_idx = 0 # Simple pagination: Just show top 20 for now as per requirement "Rank all the results, 20 per page"
            # Getting full pagination logic in Streamlit can be tricky with reruns.
            # Let's implement basic "next page" logic manually or just use slicing.
            
            # Fix: We'll just show the first page by default, and maybe a "Load More" or proper pagination controls.
            # Requirement: "Rank all the results, 20 per page starting from the highest similarity."
            
            # Let's do simple query parameter or session state for page
            # To detect query change, we can check session state
            if "last_query" not in st.session_state or st.session_state.last_query != query:
                st.session_state.page_number = 0
                st.session_state.last_query = query
            
            page = st.session_state.page_number
            start = page * ITEMS_PER_PAGE
            end = start + ITEMS_PER_PAGE
            
            page_results = results.iloc[start:end]
            
            st.write(f"Showing results {start+1}-{min(end, total_results)} of {total_results}")
            
            for _, row in page_results.iterrows():
                # Name - Website - Program Category - FA Code
                # Primary Location (City), Primary Location (Country)
                # Description
                
                with st.container(border=True):
                    header = f"**{row['Name']}** - [{row['Website']}](https://{row['Website']}) - {row['Program Category']} - {row['FA Code']}"
                    st.markdown(header)
                    st.text(f"{row.get('Primary Location (City)', '')}, {row.get('Primary Location (Country)', '')}")
                    # Map Description (SOSV) as Description
                    desc = row.get('Description (SOSV)', row.get('Description', ''))
                    st.markdown(desc)
                    st.caption(f"Similarity: {row['similarity']:.4f}")
            
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
    st.info("Enter a query to start searching.")
