import pandas as pd
import streamlit as st
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional

# System instructions for GPT
SYSTEM_INSTRUCTIONS = """
Du er en rådgiver som skal hjelpe med å kategorisere potensielle deltakere til seminarer.
Din oppgave er å:
1. Gruppere deltakerne etter type (f.eks. forskere, politikere, organisasjoner, næringsliv, etc.)
2. Liste opp gruppene på en oversiktlig måte

Bruk en kort og presis tone. Fokuser kun på å kategorisere deltakerne i relevante grupper.
"""

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
def get_embedding_cached(_text: str, api_key: str) -> Optional[List[float]]:
    """Cached version of get_embedding that doesn't use the client object"""
    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=_text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return None

@st.cache_data
def process_texts_for_embeddings(texts: List[str], api_key: str) -> List[Optional[List[float]]]:
    """Process all texts and get their embeddings with caching"""
    embeddings = []
    total = len(texts)
    
    progress_text = "Calculating embeddings..."
    progress_bar = st.progress(0, text=progress_text)
    
    for i, text in enumerate(texts):
        emb = get_embedding_cached(text, api_key)
        embeddings.append(emb)
        
        # Update progress
        progress = (i + 1) / total
        progress_bar.progress(progress, text=f"{progress_text} ({i+1}/{total})")
    
    progress_bar.empty()
    return embeddings

def find_similar_content(query_text: str, df: pd.DataFrame, cached_embeddings: List[List[float]], 
                        api_key: str, top_k: int = 5) -> List[Dict]:
    """Find similar content using pre-computed embeddings"""
    query_embedding = get_embedding_cached(query_text, api_key)
    if not query_embedding:
        return []
    
    # Calculate similarities for all entries
    similarities = []
    for emb in cached_embeddings:
        if emb:
            similarity = 1 - cosine(query_embedding, emb)
            similarities.append(similarity)
        else:
            similarities.append(0)
    
    # Create a dataframe with similarities for better handling
    similarity_df = pd.DataFrame({
        'index': range(len(similarities)),
        'similarity': similarities
    })
    
    # Get top_k most similar entries
    top_indices = similarity_df.nlargest(top_k, 'similarity')['index'].tolist()
    
    results = []
    for idx in top_indices:
        entry = df.iloc[idx]
        source_config = next(s for s in DATA_SOURCES if s["name"] == entry['source'])
        
        # Get speaker/organization info
        speakers = []
        if pd.notna(entry[source_config["speaker_column"]]):
            if '\n' in str(entry[source_config["speaker_column"]]):
                speakers = [s.strip() for s in entry[source_config["speaker_column"]].split('\n')]
            else:
                speakers = [entry[source_config["speaker_column"]].strip()]
        
        # Add result with full context
        results.append({
            'index': idx,  # Store the original index
            'speakers': speakers,
            'similarity': similarities[idx],
            'source': entry['source'],
            'context': entry[source_config["event_column"]] if source_config["event_column"] in entry else '',
            'content': entry[source_config["content_column"]] if source_config["content_column"] in entry else '',
        })
    
    return results

def get_gpt_analysis(query: str, speakers: List[Dict], api_key: str) -> Optional[str]:
    """Get GPT analysis of the suggestions"""
    try:
        client = OpenAI(api_key=api_key)
        context = f"""
Seminartema: {query}

Potensielle deltakere (sortert etter relevans):
"""
        for s in speakers:
            context += f"\n- {s['name']} (fra {s['context']})"

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": context}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Feil ved GPT-analyse: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Seminar Deltaker Forslag", page_icon="🎯", layout="wide")
    
    st.title("🎯 Seminar Deltaker Forslag")
    st.write("Beskriv seminaret ditt for å få forslag til relevante deltakere.")

    # Get API key from secrets
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
    
    # Pre-compute embeddings for all texts (cached)
    with st.spinner("Forbereder søkefunksjonalitet..."):
        cached_embeddings = process_texts_for_embeddings(df['combined_text'].tolist(), api_key)
    
    # Create three columns for input and filters
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
            value=0.5,
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
                # Filter by selected sources
                source_mask = df['source'].isin(selected_sources)
                filtered_df = df[source_mask]
                filtered_embeddings = [emb for emb, mask in zip(cached_embeddings, source_mask) if mask]
                
                results = find_similar_content(
                    query, 
                    filtered_df, 
                    filtered_embeddings,
                    api_key, 
                    top_k=num_suggestions
                )
                
if results:
                    # Process speakers with better tracking of sources
                    speakers_dict = {}
                    for result in results:
                        for speaker in result['speakers']:
                            speaker_key = f"{speaker}_{result['index']}"  # Using 'index' instead of 'original_index'
                            if speaker_key not in speakers_dict or result['similarity'] > speakers_dict[speaker_key]['similarity']:
                                speakers_dict[speaker_key] = {
                                    'name': speaker,
                                    'similarity': result['similarity'],
                                    'context': result['context'],
                                    'content': result['content'],
                                    'source': result['source']
                                }
                    
                    # Convert to list and sort
                    speakers = [info for info in speakers_dict.values() if info['similarity'] >= min_similarity]
                    speakers.sort(key=lambda x: x['similarity'], reverse=True)
  
                    for speaker in result['speakers']:
                    speaker_key = f"{speaker}_{result['original_index']}" 
                            
                    # Find this section in the main() function, around line 250-300:

                    if speakers:
                        # Get GPT analysis for categorization
                        with st.spinner("Kategoriserer deltakere..."):
                            analysis = get_gpt_analysis(query, speakers, api_key)
                            if analysis:
                                st.subheader("🏷️ Kategorisering av deltakere")
                                st.write(analysis)
                                st.divider()
                        
                        # Display detailed results
                        st.subheader(f"🎯 Fant {len(speakers)} potensielle deltakere")
                        
                        # REPLACE THIS ENTIRE FOR-LOOP SECTION:
                        for i, speaker in enumerate(speakers, 1):
                            with st.expander(
                                f"🎤 {speaker['name']} - {speaker['similarity']:.0%} relevans", 
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
                                    st.metric("Relevans", f"{speaker['similarity']:.2%}")
                                    if speaker['source'] == 'arendalsuka':
                                        st.markdown(f"[Gå til arrangement](arendalsuka.no)")  # Add actual URL pattern
                                    else:
                                        st.markdown(f"[Gå til høring](stortinget.no)")  # Add actual URL pattern

                        # Add download button
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
