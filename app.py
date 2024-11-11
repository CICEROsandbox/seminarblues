import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional

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

@st.cache_data
def load_source_data(source_config: Dict) -> Optional[pd.DataFrame]:
    """Load and prepare data from a single source"""
    try:
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
        
        return df
    except Exception as e:
        st.error(f"Error loading data from {source_config['name']}: {str(e)}")
        return None

@st.cache_data
def get_embedding(_text: str, _api_key: str) -> Optional[List[float]]:
    """Get embedding for a single text using OpenAI API with verification"""
    try:
        client = OpenAI(api_key=_api_key)
        # Preprocess text to ensure it's clean
        cleaned_text = _text.strip()
        if not cleaned_text:
            return None
            
        response = client.embeddings.create(
            input=cleaned_text,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        
        # Verify we got a valid embedding
        if not embedding or len(embedding) != 1536:  # OpenAI embeddings should be 1536-dimensional
            st.error(f"Invalid embedding received for text: {_text[:100]}...")
            return None
            
        return embedding
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return None

def find_similar_content(query_text: str, df: pd.DataFrame, cached_embeddings: List[List[float]], 
                        _api_key: str, top_k: int = 5) -> List[Dict]:
    """Find similar content with improved debugging and verification"""
    # Get query embedding
    query_embedding = get_embedding(query_text, _api_key)
    if not query_embedding:
        st.error("Failed to get query embedding")
        return []
    
    # Debug: Print query information
    st.write("Debug - Query Info:")
    st.write(f"Query text: {query_text}")
    st.write(f"Query embedding size: {len(query_embedding)}")
    
    # Calculate similarities for all entries
    similarities = []
    texts_for_debug = []
    
    for i, emb in enumerate(cached_embeddings):
        if emb and len(emb) == len(query_embedding):
            # Calculate cosine similarity
            cos_sim = 1 - cosine(query_embedding, emb)
            
            # Apply more nuanced scaling
            if cos_sim < 0.5:  # Low relevance
                scaled_similarity = cos_sim * 0.5
            elif cos_sim < 0.7:  # Medium relevance
                scaled_similarity = 0.25 + (cos_sim - 0.5) * 1.5
            else:  # High relevance
                scaled_similarity = 0.55 + (cos_sim - 0.7) * 1.5
                
            similarities.append(scaled_similarity)
            texts_for_debug.append((df.iloc[i]['combined_text'][:200], cos_sim, scaled_similarity))
        else:
            similarities.append(0)
            if emb:
                st.warning(f"Embedding size mismatch at index {i}: {len(emb)} vs {len(query_embedding)}")
    
    # Enhanced debug information
    with st.expander("Debug Information", expanded=True):
        st.write("Query:", query_text)
        st.write("Top 5 matched texts (with scores):")
        sorted_debug = sorted(texts_for_debug, key=lambda x: x[2], reverse=True)[:5]
        for text, raw_sim, scaled_sim in sorted_debug:
            st.write("\nText:", text)
            st.write(f"Raw similarity: {raw_sim:.3f}")
            st.write(f"Scaled similarity: {scaled_sim:.3f}")
            st.write("---")
    
    # Create similarity DataFrame
    similarity_df = pd.DataFrame({
        'index': range(len(similarities)),
        'similarity': similarities
    })
    
    # Filter low-relevance results
    threshold = 0.3
    similarity_df = similarity_df[similarity_df['similarity'] > threshold]
    
    # Get top_k matches
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
        
        final_similarity = similarities[idx]
        
        results.append({
            'index': idx,
            'speakers': speakers,
            'similarity': final_similarity,
            'source': entry['source'],
            'context': entry.get(source_config["event_column"], ''),
            'content': entry.get(source_config["content_column"], ''),
        })
    
    return results

@st.cache_data
def process_texts_for_embeddings(texts: List[str], _api_key: str) -> List[Optional[List[float]]]:
    """Process texts with improved error checking and debugging"""
    embeddings = []
    total = len(texts)
    
    progress_text = "Calculating embeddings..."
    progress_bar = st.progress(0, text=progress_text)
    
    # Debug counters
    successful = 0
    failed = 0
    
    for i, text in enumerate(texts):
        emb = get_embedding(text, _api_key)
        if emb:
            successful += 1
            embeddings.append(emb)
        else:
            failed += 1
            embeddings.append(None)
            st.warning(f"Failed to get embedding for text: {text[:100]}...")
        
        # Update progress
        progress = (i + 1) / total
        progress_bar.progress(progress, text=f"{progress_text} ({i+1}/{total})")
    
    progress_bar.empty()
    
    # Show embedding statistics
    st.write(f"Embeddings processed: {successful} successful, {failed} failed")
    
    return embeddings

def find_similar_content(query_text: str, df: pd.DataFrame, cached_embeddings: List[List[float]], 
                        _api_key: str, top_k: int = 5) -> List[Dict]:
    """Find similar content using pre-computed embeddings with realistic similarity scoring"""
    query_embedding = get_embedding(query_text, _api_key)
    if not query_embedding:
        return []
    
    # Calculate similarities for all entries
    similarities = []
    texts_for_debug = []
    
    for i, emb in enumerate(cached_embeddings):
        if emb:
            # Calculate raw cosine similarity
            cos_sim = 1 - cosine(query_embedding, emb)
            
            # Apply more realistic scaling:
            # 1. Start with raw cosine similarity
            # 2. Apply sigmoid-like transformation to spread out the middle range
            # 3. Penalize low similarities more aggressively
            
            # First, center the similarity scores around 0.5
            centered_sim = cos_sim - 0.5
            
            # Apply sigmoid-like transformation
            if centered_sim > 0:
                scaled_similarity = 0.5 + (centered_sim * 0.8)  # Reduce upper range
            else:
                scaled_similarity = 0.5 + (centered_sim * 2.0)  # Increase lower range penalty
            
            # Additional penalty for very low similarities
            if cos_sim < 0.7:  # Higher threshold for relevance
                scaled_similarity *= (cos_sim / 0.7)  # Progressive penalty
            
            # Ensure we stay within 0-1 range
            scaled_similarity = max(0.0, min(1.0, scaled_similarity))
            
            similarities.append(scaled_similarity)
            texts_for_debug.append((df.iloc[i]['combined_text'][:200], cos_sim, scaled_similarity))
        else:
            similarities.append(0)
    
    # Enhanced debug information
    with st.expander("Debug Information", expanded=False):
        st.write("Top 3 matched texts (with detailed scoring):")
        sorted_debug = sorted(texts_for_debug, key=lambda x: x[2], reverse=True)[:3]
        for text, raw_sim, scaled_sim in sorted_debug:
            st.write(f"Raw similarity: {raw_sim:.3f}")
            st.write(f"Scaled similarity: {scaled_sim:.3f}")
            st.write(f"Text: {text}...")
            st.write("---")
    
    # Use stricter filtering
    similarity_df = pd.DataFrame({
        'index': range(len(similarities)),
        'similarity': similarities
    })
    
    # Much stricter threshold for filtering
    min_similarity = 0.4  # Base threshold
    similarity_df = similarity_df[similarity_df['similarity'] > min_similarity]
    
    # Get top_k matches
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
        
        # Apply final percentage scaling
        final_similarity = similarities[idx]
        # Convert to a more realistic percentage range
        percentage_similarity = final_similarity * 100
        
        results.append({
            'index': idx,
            'speakers': speakers,
            'similarity': percentage_similarity / 100,  # Keep as decimal but with realistic scaling
            'source': entry['source'],
            'context': entry.get(source_config["event_column"], ''),
            'content': entry.get(source_config["content_column"], ''),
        })
    
    return results

def main():
    st.set_page_config(page_title="Seminar Deltaker Forslag", page_icon="🎯", layout="wide")
    
    st.title("🎯 Seminar Deltaker Forslag")
    st.write("Beskriv seminaret ditt for å få forslag til relevante deltakere.")

    # Get API key from secrets
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OpenAI API key not found in secrets!")
        st.stop()
    
    api_key = st.secrets["OPENAI_API_KEY"]

    # Load data from all sources
    all_data = []
    with st.spinner("Laster inn data..."):
        for source_config in DATA_SOURCES:
            df = load_source_data(source_config)
            if df is not None:
                all_data.append(df)
    
    if not all_data:
        st.error("Kunne ikke laste inn data. Sjekk datakildene.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    
    # Pre-compute embeddings
    with st.spinner("Forbereder søkefunksjonalitet..."):
        cached_embeddings = process_texts_for_embeddings(df['combined_text'].tolist(), api_key)
    
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
