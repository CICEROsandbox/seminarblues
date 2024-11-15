import streamlit as st
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding  # Requires OpenAI API
from scipy.spatial.distance import cosine
import hashlib
import os
import glob
import re
import time
from typing import List, Dict, Tuple, Optional, Set

# Configuration
DATA_DIRECTORY = "data"
EMBEDDING_MODEL = "text-embedding-ada-002"
TEXT_COLUMN = "description"  # Adjust to your specific column name if needed

st.set_page_config(page_title="Seminar Deltaker Forslag", page_icon="🎯", layout="wide")

# Define stop words, keywords, etc. as in your original code
NORWEGIAN_STOP_WORDS = {...}
CLIMATE_KEYWORDS = {...}
CLIMATE_CATEGORIES = {...}

def hash_file(file_path: str) -> str:
    """Generate a hash for a file to check for changes."""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def generate_embeddings_cache(file_path: str, text_column: str = TEXT_COLUMN, model: str = EMBEDDING_MODEL):
    """Generate embeddings for a text column in a CSV file and cache them."""
    file_hash = hash_file(file_path)
    cache_file = f"{file_path}_{file_hash}_embeddings.npy"
    
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings for {file_path}...")
        embeddings = np.load(cache_file, allow_pickle=True)
    else:
        print(f"Generating embeddings for {file_path}...")
        data = pd.read_csv(file_path)
        texts = data[text_column].fillna("").tolist()
        embeddings = np.array([get_embedding(text, model=model) for text in texts])
        np.save(cache_file, embeddings)
        print(f"Cached embeddings saved to {cache_file}.")
    
    return embeddings

def load_all_embeddings(data_directory: str = DATA_DIRECTORY, text_column: str = TEXT_COLUMN) -> Dict[str, np.ndarray]:
    """Load embeddings for all CSV files in the specified directory."""
    embeddings_dict = {}
    
    for file_path in glob.glob(os.path.join(data_directory, "*.csv")):
        try:
            embeddings = generate_embeddings_cache(file_path, text_column)
            embeddings_dict[file_path] = embeddings
        except Exception as e:
            st.error(f"Error processing {file_path}: {e}")
    
    return embeddings_dict

# Load all cached embeddings at the start of the app
all_embeddings = load_all_embeddings()

@st.cache_data(ttl=300)
def get_embedding_for_text(text: str) -> Optional[List[float]]:
    """Get embedding for a single text for query processing."""
    try:
        response = openai.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return response['data'][0]['embedding']
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return None

def calculate_similarity(query_embedding: List[float], doc_embedding: List[float]) -> float:
    """Calculate cosine similarity between query and document embeddings."""
    return 1 - cosine(query_embedding, doc_embedding)

def main():
    st.title("🎯 Seminar Deltaker Forslag")
    st.write("Beskriv seminaret ditt for å få forslag til relevante deltakere.")
    
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OpenAI API key not found in secrets!")
        st.stop()
    
    api_key = st.secrets["OPENAI_API_KEY"]
    query = st.text_area("Beskriv seminar-temaet:", height=100, placeholder="Eksempel: Et seminar om klimatilpasning...")

    if st.button("Finn deltakere"):
        if query:
            with st.spinner("Søker etter relevante deltakere..."):
                query_embedding = get_embedding_for_text(query)
                if not query_embedding:
                    st.error("Failed to retrieve query embedding.")
                    return
                
                results = []
                for file_path, embeddings in all_embeddings.items():
                    data = pd.read_csv(file_path)
                    for i, doc_embedding in enumerate(embeddings):
                        similarity = calculate_similarity(query_embedding, doc_embedding)
                        if similarity > 0.3:  # Filter for minimum similarity threshold
                            results.append({
                                "source": file_path,
                                "similarity": similarity,
                                "content": data.iloc[i][TEXT_COLUMN]
                            })
                
                results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:5]  # Top 5 results
                
                if results:
                    st.subheader("Forslag til deltakere:")
                    for result in results:
                        st.write(f"**{result['source']}** - Relevans: {result['similarity']:.2f}")
                        st.write(result["content"])
                else:
                    st.warning("Fant ingen relevante forslag.")
                    
if __name__ == "__main__":
    main()
