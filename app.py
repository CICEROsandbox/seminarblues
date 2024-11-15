import streamlit as st
import pandas as pd
import numpy as np
import openai
from scipy.spatial.distance import cosine
import hashlib
import os
import glob
import re
from typing import List, Dict, Tuple, Set, Optional

# Configuration
DATA_DIRECTORY = "data"
EMBEDDING_MODEL = "text-embedding-ada-002"
DATA_SOURCES = [
    {"file_path": "data/arendalsuka_events.csv", "text_columns": ["title", "header", "om_arrangementet"], "separator": ","},
    {"file_path": "data/arendalsuka_events_klima.csv", "text_columns": ["title", "header", "om_arrangementet"], "separator": ","},
    {"file_path": "data/stortinget-hearings.csv", "text_columns": ["Høringssak", "Innhold - høring"], "separator": ";"}
]

st.set_page_config(page_title="Seminar Deltaker Forslag", page_icon="🎯", layout="wide")

# Define stop words, keywords, etc.
NORWEGIAN_STOP_WORDS = {...}  # Keep your full set here
CLIMATE_KEYWORDS = {...}  # Keep your full set here

@st.cache_data(ttl=300)
def get_embedding_for_text(text: str) -> Optional[List[float]]:
    """Get embedding for a single text for query processing."""
    try:
        response = openai.Embedding.create(input=text, model=EMBEDDING_MODEL)
        return response['data'][0]['embedding']
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return None

def hash_file(file_path: str) -> str:
    """Generate a hash for a file to check for changes."""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def generate_embeddings_cache(file_path: str, text_columns: List[str], separator: str = ",", model: str = EMBEDDING_MODEL):
    """Generate embeddings for combined text columns in a CSV file and cache them."""
    file_hash = hash_file(file_path)
    cache_file = f"{file_path}_{file_hash}_embeddings.npy"
    
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings for {file_path}...")
        embeddings = np.load(cache_file, allow_pickle=True)
    else:
        print(f"Generating embeddings for {file_path}...")
        try:
            data = pd.read_csv(file_path, sep=separator)
            for col in text_columns:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in {file_path}.")
            
            # Combine the specified text columns for embedding
            data["combined_text"] = data[text_columns].fillna('').agg(' '.join, axis=1)
            texts = data["combined_text"].tolist()
            
            embeddings = np.array([get_embedding_for_text(text) for text in texts])
            np.save(cache_file, embeddings)
            print(f"Cached embeddings saved to {cache_file}.")
        except Exception as e:
            st.error(f"Error processing {file_path}: {e}")
            embeddings = None
    
    return embeddings

def load_all_embeddings(data_sources: List[Dict]) -> Dict[str, np.ndarray]:
    """Load embeddings for all CSV files specified in data sources."""
    embeddings_dict = {}
    
    for source in data_sources:
        file_path = source["file_path"]
        text_columns = source["text_columns"]
        separator = source.get("separator", ",")
        
        embeddings = generate_embeddings_cache(file_path, text_columns, separator)
        if embeddings is not None:
            embeddings_dict[file_path] = embeddings
    
    return embeddings_dict

# Load all cached embeddings at the start of the app
all_embeddings = load_all_embeddings(DATA_SOURCES)

def extract_keywords_from_text(text: str) -> Set[str]:
    """Extract all meaningful words from text, excluding stop words."""
    words = re.findall(r'\w+', text.lower())
    keywords = {word for word in words if word not in NORWEGIAN_STOP_WORDS and len(word) > 2}
    return keywords

def calculate_similarity(query_embedding: List[float], doc_embedding: List[float], boost_keywords: Set[str], doc_text: str) -> float:
    """Calculate a custom similarity score with emphasis on boosted keywords."""
    base_similarity = 1 - cosine(query_embedding, doc_embedding)
    doc_keywords = extract_keywords_from_text(doc_text)
    keyword_match_score = len(doc_keywords.intersection(boost_keywords)) / (len(boost_keywords) + 1)
    return 0.7 * base_similarity + 0.3 * keyword_match_score

def main():
    st.title("🎯 Seminar Deltaker Forslag")
    st.write("Beskriv seminaret ditt for å få forslag til relevante deltakere.")
    
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OpenAI API key not found in secrets!")
        st.stop()
    
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    query = st.text_area("Beskriv seminar-temaet:", height=100, placeholder="Eksempel: Et seminar om klimatilpasning...")
    boost_keywords = extract_keywords_from_text(query)
    
    if st.button("Finn deltakere"):
        if query:
            with st.spinner("Søker etter relevante deltakere..."):
                query_embedding = get_embedding_for_text(query)
                if not query_embedding:
                    st.error("Failed to retrieve query embedding.")
                    return
                
                results = []
                for source in DATA_SOURCES:
                    file_path = source["file_path"]
                    text_columns = source["text_columns"]
                    if file_path in all_embeddings:
                        data = pd.read_csv(file_path, sep=source.get("separator", ","))
                        data["combined_text"] = data[text_columns].fillna('').agg(' '.join, axis=1)
                        for i, doc_embedding in enumerate(all_embeddings[file_path]):
                            similarity = calculate_similarity(query_embedding, doc_embedding, boost_keywords, data.iloc[i]["combined_text"])
                            if similarity > 0.3:
                                results.append({
                                    "source": file_path,
                                    "similarity": similarity,
                                    "content": data.iloc[i]["combined_text"]
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
