import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)

INPUT_FILE = "portfolio.csv"
OUTPUT_FILE = "embed.csv"

def generate_embeddings():
    if os.path.exists(OUTPUT_FILE):
        print(f"{OUTPUT_FILE} already exists. Skipping generation.")
        return

    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    if "Pre-Embed" not in df.columns:
        print("Error: 'Pre-Embed' column not found in input file.")
        return

    print("Generating embeddings with batching...")
    
    embeddings = []
    batch_size = 50 # Safe batch size
    texts = df["Pre-Embed"].astype(str).tolist()
    total = len(texts)
    
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i+batch_size]
        # Clean inputs: replace newlines or non-string just in case, though astype(str) helps.
        # Gemini sometimes dislikes empty strings in batch?
        batch_texts = [t if t.strip() else " " for t in batch_texts]
        
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=batch_texts,
                task_type="retrieval_document",
                title="Portfolio Entry"
            )
            # Result is expected to be a dict with key 'embedding' which is a list of lists
            batch_embeddings = result['embedding']
            embeddings.extend(batch_embeddings)
            print(f"Processed {min(i + batch_size, total)}/{total}")
            time.sleep(0.5) # Rate limit protection
            
        except Exception as e:
            print(f"Error embedding batch starting at {i}: {e}")
            # Fallback: append None for this batch so indices match
            embeddings.extend([None] * len(batch_texts))

    if len(embeddings) != len(df):
        print(f"Warning: Embedding count {len(embeddings)} does not match row count {len(df)}")
        # Resize to match
        if len(embeddings) < len(df):
             embeddings.extend([None] * (len(df) - len(embeddings)))
        else:
             embeddings = embeddings[:len(df)]

    df["embedding"] = embeddings
    
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    generate_embeddings()
