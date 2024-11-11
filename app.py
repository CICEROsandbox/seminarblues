import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional
import time

# Configuration
DATA_SOURCES = [
    {
        "name": "arendalsuka",
        "file_path": "data/arendalsuka_events.csv",
        "text_columns": ['title', 'header', 'om_arrangementet'],
        "speaker_column": 'medvirkende',
        "date_column": 'date',
        "event_column": 'title',
        "content_column": 'om_arrangementet'
    },
    {
        "name": "parliament_hearings",
        "file_path": "data/stortinget-hearings.csv",
        "text_columns": ['Høringssak', 'Innhold - høring'],
        "speaker_column": 'Innsender',
        "event_column": 'Høringssak',
        "content_column": 'Innhold - høring',
        "separator": ";"
    }
]

@st.cache_data(show_spinner=True)
def load_source_data(source_config: Dict) -> Optional[pd.DataFrame]:
    """Load and prepare data from a single source"""
    try:
        st.write(f"Loading data from {source_config['name']}...")
        separator = source_config.get("separator", ",")
        df = pd.read_csv(source_config["file_path"], sep=separator)
        df = df.dropna(how='all')
        
        # Combine specified text columns for embedding
        df['combined_text'] = ''
        for col in source_config["text_columns"]:
            if col in df.columns:
                df['combined_text'] += ' ' + df[col].fillna('')
        
        df['combined_text'] = df['combined_text'].str.strip()
        df['source'] = source_config["name"]
        
        st.write(f"Loaded {len(df)} rows from {source_config['name']}")
        return df
    except Exception as e:
        st.error(f"Error loading data from {source_config['name']}: {str(e)}")
        return None

@st.cache_data(ttl=300)
def get_embedding(_text: str, _api_key: str) -> Optional[List[float]]:
    """Get embedding for a single text"""
    try:
        client = OpenAI(api_key=_api_key)
        response = client.embeddings.create(
            input=_text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return None

def calculate_similarity(query_embedding: List[float], doc_embedding: List[float]) -> float:
    """Calculate semantic similarity with proper scaling"""
    if not query_embedding or not doc_embedding:
        return 0.0
    
    # Calculate base similarity
    cos_sim = 1 - cosine(query_embedding, doc_embedding)
    
    # Apply non-linear scaling to better differentiate relevance levels
    if cos_sim < 0.5:
        return cos_sim * 0.2  # Very low relevance
    elif cos_sim < 0.7:
        return 0.1 + (cos_sim - 0.5) * 0.6  # Medium relevance
    else:
        return 0.22 + (cos_sim - 0.7) * 1.2  # High relevance

@st.cache_data
def process_texts_for_embeddings(texts: List[str], _api_key: str) -> List[Optional[List[float]]]:
    """Process texts for embeddings with better progress tracking"""
    embeddings = []
    total = len(texts)
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, text in enumerate(texts):
        try:
            status_text.text(f"Processing document {i+1} of {total}")
            emb = get_embedding(text, _api_key)
            embeddings.append(emb)
            progress_bar.progress((i + 1) / total)
            
            # Add a small delay to prevent rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            st.error(f"Error processing document {i+1}: {str(e)}")
            embeddings.append(None)
    
    progress_bar.empty()
    status_text.empty()
    
    return embeddings

def find_similar_content(query_text: str, df: pd.DataFrame, cached_embeddings: List[List[float]], 
                        _api_key: str, top_k: int = 5) -> List[Dict]:
    """Find similar content without caching similarity calculations"""
    query_embedding = get_embedding(query_text, _api_key)
    if not query_embedding:
        return []
    
    # Calculate similarities fresh each time
    similarities = []
    texts_for_debug = []
    
    for i, emb in enumerate(cached_embeddings):
        if emb:
            similarity = calculate_similarity(query_embedding, emb)
            similarities.append(similarity)
            texts_for_debug.append((df.iloc[i]['combined_text'][:200], similarity))
        else:
            similarities.append(0)
    
    # Debug information
    with st.expander("Search Analysis", expanded=True):
        st.write(f"Query: '{query_text}'")
        st.write("Top matches:")
        sorted_debug = sorted(texts_for_debug, key=lambda x: x[1], reverse=True)[:3]
        for text, sim in sorted_debug:
            st.write(f"\nScore: {sim:.3f}")
            st.write(f"Text: {text}")
            st.write("---")
    
    # Filter and rank results
    similarity_df = pd.DataFrame({
        'index': range(len(similarities)),
        'similarity': similarities
    })
    
    # Filter low relevance matches
    similarity_df = similarity_df[similarity_df['similarity'] > 0.3]
    top_indices = similarity_df.nlargest(top_k, 'similarity')['index'].tolist()
    
    results = []
    for idx in top_indices:
        entry = df.iloc[idx]
        source_config = next(s for s in DATA_SOURCES if s["name"] == entry['source'])
        
        speakers = []
        if pd.notna(entry[source_config["speaker_column"]]):
            if '\n' in str(entry[source_config["speaker_column"]]):
                speakers = [s.strip() for s in entry[source_config["speaker_column"]].split('\n')]
            else:
                speakers = [entry[source_config["speaker_column"]].strip()]
        
        results.append({
            'index': idx,
            'speakers': speakers,
            'similarity': float(similarities[idx]),
            'source': entry['source'],
            'context': entry.get(source_config["event_column"], ''),
            'content': entry.get(source_config["content_column"], ''),
        })
    
    return results

def main():
    st.set_page_config(page_title="Seminar Deltaker Forslag", page_icon="🎯", layout="wide")
    
    # Add initialization status
    status = st.empty()
    status.text("Initializing application...")
    
    st.title("🎯 Seminar Deltaker Forslag")
    st.write("Beskriv seminaret ditt for å få forslag til relevante deltakere.")
    
    # Add cache clear button
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.rerun()

    # Check for API key
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OpenAI API key not found in secrets!")
        st.stop()
    
    api_key = st.secrets["OPENAI_API_KEY"]
    
    # Load data with progress tracking
    status.text("Loading data sources...")
    all_data = []
    for source_config in DATA_SOURCES:
        df = load_source_data(source_config)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        st.error("Could not load any data sources. Please check the data files.")
        st.stop()
    
    status.text("Combining data sources...")
    df = pd.concat(all_data, ignore_index=True)
    st.write(f"Total records loaded: {len(df)}")
    
    # Process embeddings
    status.text("Processing document embeddings...")
    cached_embeddings = process_texts_for_embeddings(df['combined_text'].tolist(), api_key)
    st.write(f"Processed {len(cached_embeddings)} embeddings")
    
    status.empty()
    
    # Create input layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        query = st.text_area(
            "Beskriv seminar-temaet:",
            height=100,
            placeholder="Eksempel: Et seminar om karbonfangst og lagring i Norge, med fokus på politiske rammeverk og industrisamarbeid."
        )
    
    with col2:
        num_suggestions = st.slider(
            "Antall forslag å vurdere:",
            min_value=3,
            max_value=15,
            value=5
        )
        
        min_similarity = st.slider(
            "Minimum relevans (0-1):",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05
        )
    
    with col3:
        selected_sources = st.multiselect(
            "Filtrer kilder:",
            options=[source["name"] for source in DATA_SOURCES],
            default=[source["name"] for source in DATA_SOURCES],
            format_func=lambda x: "Arendalsuka" if x == "arendalsuka" else "Stortingshøringer"
        )

    if st.button("Finn deltakere", type="primary"):
        if query:
            with st.spinner("Søker etter relevante deltakere..."):
                source_mask = df['source'].isin(selected_sources)
                filtered_df = df[source_mask].reset_index(drop=True)
                filtered_embeddings = [emb for emb, mask in zip(cached_embeddings, source_mask) if mask]
                
                results = find_similar_content(
                    query, 
                    filtered_df, 
                    filtered_embeddings,
                    api_key, 
                    top_k=num_suggestions
                )
                
                if results:
                    speakers_dict = {}
                    for result in results:
                        for speaker in result['speakers']:
                            speaker_key = f"{speaker}_{result['index']}"
                            if speaker_key not in speakers_dict or result['similarity'] > speakers_dict[speaker_key]['similarity']:
                                speakers_dict[speaker_key] = {
                                    'name': speaker,
                                    'similarity': result['similarity'],
                                    'context': result['context'],
                                    'content': result['content'],
                                    'source': result['source']
                                }
                    
                    speakers = [info for info in speakers_dict.values() if info['similarity'] >= min_similarity]
                    speakers.sort(key=lambda x: x['similarity'], reverse=True)
                    
                    if speakers:
                        st.subheader(f"🎯 Fant {len(speakers)} potensielle deltakere")
                        
                        for i, speaker in enumerate(speakers, 1):
                            with st.expander(
                                f"🎤 {speaker['name']} - {speaker['similarity']:.1%} relevans", 
                                expanded=i<=3
                            ):
                                cols = st.columns([2, 1])
                                with cols[0]:
                                    if speaker['source'] == 'arendalsuka':
                                        st.write("**Deltaker i arrangement:**", speaker['context'])
                                        if pd.notna(speaker['content']):
                                            st.write("**Arrangementsbeskrivelse:**")
                                            st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>{speaker['content']}</div>", unsafe_allow_html=True)
                                    else:  # parliament hearings
                                        st.write("**Innspill til høring:**", speaker['context'])
                                        if pd.notna(speaker['content']):
                                            st.write("**Høringsinnspill:**")
                                            st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>{speaker['content']}</div>", unsafe_allow_html=True)
                                    
                                    st.write("**Kilde:**", 
                                           "Arendalsuka" if speaker['source'] == "arendalsuka" 
                                           else "Stortingshøringer")
                                with cols[1]:
                                    st.metric("Relevans", f"{speaker['similarity']:.1%}")
                                    if speaker['source'] == 'arendalsuka':
                                        st.markdown("[Gå til arrangement](https://arendalsuka.no)")
                                    else:
                                        st.markdown("[Gå til høring](https://stortinget.no)")
                        
                        st.download_button(
                            "Last ned forslag som CSV",
                            pd.DataFrame(speakers).to_csv(index=False),
                            "deltaker_forslag.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    else:
                        st.warning("Ingen deltakere møtte minimumskravet for relevans. Prøv å justere filteret.")
                else:
                    st.warning("Fant ingen relevante forslag. Prøv å justere søkekriteriene.")
        else:
            st.warning("Vennligst beskriv seminar-temaet.")

if __name__ == "__main__":
    main()
