import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional, Set, Tuple
import time

# Add the stop words list here, right after the imports and before DATA_SOURCES:
NORWEGIAN_STOP_WORDS = {
    'og', 'i', 'jeg', 'det', 'at', 'en', 'et', 'den', 'til', 'er', 'som', 'på',
    'de', 'med', 'han', 'av', 'ikke', 'der', 'så', 'var', 'meg', 'seg', 'men',
    'ett', 'har', 'om', 'vi', 'min', 'mitt', 'ha', 'hadde', 'hun', 'nå', 'over',
    'da', 'ved', 'fra', 'du', 'ut', 'sin', 'dem', 'oss', 'opp', 'man', 'kan',
    'hans', 'hvor', 'eller', 'hva', 'skal', 'selv', 'sjøl', 'her', 'alle',
    'vil', 'bli', 'ble', 'blitt', 'kunne', 'inn', 'når', 'være', 'kom', 'noen',
    'noe', 'ville', 'dere', 'som', 'deres', 'kun', 'ja', 'etter', 'ned', 'skulle',
    'denne', 'for', 'deg', 'si', 'sine', 'sitt', 'mot', 'å', 'meget', 'hvorfor',
    'dette', 'disse', 'uten', 'hvordan', 'ingen', 'din', 'ditt', 'blir', 'samme',
    'hvilken', 'hvilke', 'sånn', 'inni', 'mellom', 'vår', 'hver', 'hvem', 'vors',
    'hvis', 'både', 'bare', 'enn', 'fordi', 'før', 'mange', 'også', 'slik',
    'vært', 'være', 'begge', 'siden', 'henne', 'hennar', 'hennes', 'lære', 'på',
    'med', 'hverandre'
}

CLIMATE_KEYWORDS = {
    'aksept', 'arealbruk', 'arktis', 'co2', 'utslipp', 'ekstremvær', 
    'energiforbruk', 'energipolitikk', 'flom', 'klimapanel', 'forbruk',
    'fornybar', 'energi', 'klima', 'helse', 'hetebølge', 'hydrogen',
    'karbon', 'karbonfangst', 'klimabudsjett', 'klimafinans', 
    'klimaforhandling', 'klimakommunikasjon', 'klimamodell', 'klimaomstilling',
    'klimapolitikk', 'klimarisiko', 'klimatjeneste', 'luftforurensning',
    'landbruk', 'metan', 'nedbør', 'olje', 'gass', 'atmosfære', 'omstilling',
    'sirkulærøkonomi', 'skog', 'teknologi', 'temperatur', 'tilpasning',
    'transport', 'utslipp', 'vindkraft', 'klimaendring'
}

CLIMATE_CATEGORIES = {
    'Klimaendringer': ['klimaendring', 'ekstremvær', 'hetebølger', 'flom', 'temperaturendringer', 'nedbørsendringer'],
    'Energi': ['energiforbruk', 'fornybar energi', 'hydrogen', 'vindkraft', 'energipolitikk'],
    'Utslipp': ['co2-utslipp', 'metan', 'karbonbudsjett', 'utslippsscenarier'],
    'Politikk og Omstilling': ['klimapolitikk', 'klimaforhandlingene', 'rettferdig omstilling', 'klimaomstilling'],
    'Miljø og Ressurser': ['arealbruk', 'skog', 'sirkulærøkonomi', 'mat og landbruk'],
    'Tilpasning og Risiko': ['klimarisiko', 'tilpasning', 'klimatjenester'],
    'Samfunn og Helse': ['helse', 'luftforurensning', 'klimakommunikasjon', 'aksept']
}

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

def extract_relevant_keywords(text: str, categories: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Extract potentially relevant keywords from text based on categories"""
    text_lower = text.lower()
    relevant_keywords = {}
    
    for category, keywords in categories.items():
        matches = []
        for keyword in keywords:
            # Check for both exact and partial matches
            if keyword.lower() in text_lower or any(
                term in text_lower for term in keyword.lower().split()
            ):
                matches.append(keyword)
        if matches:
            relevant_keywords[category] = matches
    
    return relevant_keywords

def render_keyword_selection(suggested_keywords: Dict[str, List[str]], key_prefix: str = "") -> Set[str]:
    """Render interactive keyword selection interface"""
    selected_keywords = set()
    
    st.write("#### 🏷️ Foreslåtte nøkkelord")
    st.write("Klikk for å fjerne irrelevante nøkkelord:")
    
    for category, keywords in suggested_keywords.items():
        if keywords:  # Only show categories with matches
            st.write(f"**{category}:**")
            cols = st.columns(4)  # Adjust number of columns based on your needs
            for idx, keyword in enumerate(keywords):
                col_idx = idx % 4
                with cols[col_idx]:
                    if st.button(
                        f"❌ {keyword}",
                        key=f"{key_prefix}_{category}_{keyword}",
                        type="secondary",
                        use_container_width=True
                    ):
                        # Button click means remove/ignore this keyword
                        pass
                    else:
                        selected_keywords.add(keyword)
    
    return selected_keywords
    
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

def calculate_similarity(
    query_embedding: List[float],
    doc_embedding: List[float],
    query_text: str,
    doc_text: str,
    boost_keywords: Set[str] = None
) -> Tuple[float, Set[str]]:
    """Calculate semantic similarity with emphasis on climate research relevance and selected keywords"""
    if not query_embedding or not doc_embedding:
        return 0.0, set()
    
    # Base similarity calculation
    cos_sim = 1 - cosine(query_embedding, doc_embedding)
    
    # Text preprocessing
    query_lower = query_text.lower()
    doc_lower = doc_text.lower()
    
    # Word matching calculations
    query_words = {word for word in query_lower.split() 
                  if word not in NORWEGIAN_STOP_WORDS and len(word) > 2}
    doc_words = {word for word in doc_lower.split() 
                if word not in NORWEGIAN_STOP_WORDS and len(word) > 2}
    
    # Find matching words
    matching_words = query_words.intersection(doc_words)
    
    # Calculate climate keyword relevance
    climate_words_doc = {word for word in doc_words if any(
        climate_term in word for climate_term in CLIMATE_KEYWORDS
    )}
    
    # Selected keywords boost
    selected_keyword_matches = 0
    if boost_keywords:
        selected_keyword_matches = sum(
            1 for keyword in boost_keywords
            if any(keyword.lower() in word.lower() for word in doc_words)
        )
        selected_keyword_ratio = selected_keyword_matches / len(boost_keywords)
    else:
        selected_keyword_ratio = 0
    
    # Calculate final score with components
    final_score = (
        0.50 * max(0, cos_sim) +                # Base semantic similarity
        0.20 * selected_keyword_ratio +         # Selected keywords boost
        0.20 * len(climate_words_doc) / 10 +    # Climate terms presence
        0.10 * len(matching_words) / len(query_words) if query_words else 0  # General relevance
    )
    
    # Apply threshold adjustments
    if final_score < 0.2:
        final_score *= 0.5
    elif final_score > 0.8:
        final_score = 0.8 + (final_score - 0.8) * 0.5
    
    return final_score, matching_words

def find_similar_content(query_text: str, df: pd.DataFrame, cached_embeddings: List[List[float]], 
                        api_key: str, top_k: int = 5, boost_keywords: Set[str] = None) -> List[Dict]:
    """Find similar content with climate research context"""
    # Add climate research context to the query
    climate_context = "I kontekst av klimaforskning, energi, og bærekraftig omstilling: "
    enhanced_query = climate_context + query_text
    
    query_embedding = get_embedding(enhanced_query, api_key)
    if not query_embedding:
        return []
    
    # Calculate similarities
    similarities = []
    texts_for_debug = []
    
    for i, emb in enumerate(cached_embeddings):
        if emb:
            similarity, matching_words = calculate_similarity(
                query_embedding, 
                emb,
                query_text,
                df.iloc[i]['combined_text']
            )
            
            similarities.append({
                'index': i,
                'score': similarity,
                'source': df.iloc[i]['source'],
                'matching_words': matching_words
            })
            
            texts_for_debug.append((
                df.iloc[i]['combined_text'][:200],
                similarity,
                df.iloc[i]['source'],
                matching_words
            ))
        else:
            similarities.append({
                'index': i,
                'score': 0,
                'source': df.iloc[i]['source'],
                'matching_words': set()
            })
    
    # Enhanced debug information
    with st.expander("Search Analysis", expanded=False):
        st.write(f"Query: '{query_text}'")
        st.write("Top matches:")
        sorted_debug = sorted(texts_for_debug, key=lambda x: x[1], reverse=True)[:5]
        for text, sim, source, matching_words in sorted_debug:
            st.write("\n---")
            st.write(f"Source: {source}")
            st.write(f"Score: {sim:.3f}")
            if matching_words:
                st.write(f"Matching words: {', '.join(matching_words)}")
            st.write(f"Text: {text}")
    
    # Filter and group results by source
    max_per_source = min(3, top_k)  # Maximum results per source
    results_by_source = {}
    
    # Sort similarities by score
    sorted_similarities = sorted(
        [s for s in similarities if isinstance(s, dict) and s['score'] > 0.2],  # Minimum threshold
        key=lambda x: x['score'],
        reverse=True
    )
    
    # Group by source while respecting max_per_source
    for sim in sorted_similarities:
        source = sim['source']
        if source not in results_by_source:
            results_by_source[source] = []
        if len(results_by_source[source]) < max_per_source:
            results_by_source[source].append(sim)
    
    # Prepare final results
    results = []
    for source_results in results_by_source.values():
        for sim in source_results:
            idx = sim['index']
            entry = df.iloc[idx]
            source_config = next(s for s in DATA_SOURCES if s["name"] == entry['source'])
            
            # Process speakers
            speakers = []
            if pd.notna(entry[source_config["speaker_column"]]):
                if '\n' in str(entry[source_config["speaker_column"]]):
                    speakers = [s.strip() for s in entry[source_config["speaker_column"]].split('\n')]
                else:
                    speakers = [entry[source_config["speaker_column"]].strip()]
            
            results.append({
                'index': idx,
                'speakers': speakers,
                'similarity': float(sim['score']),
                'source': entry['source'],
                'context': entry.get(source_config["event_column"], ''),
                'content': entry.get(source_config["content_column"], ''),
                'matching_words': list(sim['matching_words'])
            })
    
    # Limit total results while maintaining source balance
    return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]

@st.cache_data
def process_texts_for_embeddings(texts: List[str], _api_key: str) -> List[Optional[List[float]]]:
    """Process texts for embeddings with progress tracking"""
    embeddings = []
    total = len(texts)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, text in enumerate(texts):
        try:
            status_text.text(f"Processing document {i+1} of {total}")
            emb = get_embedding(text, _api_key)
            embeddings.append(emb)
            progress_bar.progress((i + 1) / total)
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            st.error(f"Error processing document {i+1}: {str(e)}")
            embeddings.append(None)
    
    progress_bar.empty()
    status_text.empty()
    st.write(f"Processed {len(embeddings)} embeddings")
    
    return embeddings

def main():
    st.set_page_config(page_title="Seminar Deltaker Forslag", page_icon="🎯", layout="wide")
    
    st.title("🎯 Seminar Deltaker Forslag")
    st.write("Beskriv seminaret ditt for å få forslag til relevante deltakere.")
    
    # Add cache clear button
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.rerun()
    
    # Add topic guidance
    st.info("""
    💡 Tips for bedre resultater:
    - Vær spesifikk om temaet
    - Inkluder nøkkelord og konsepter
    - Beskriv formålet med seminaret
    """)

    # Check for API key
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OpenAI API key not found in secrets!")
        st.stop()
    
    api_key = st.secrets["OPENAI_API_KEY"]
    
    # Load data
    all_data = []
    for source_config in DATA_SOURCES:
        df = load_source_data(source_config)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        st.error("Could not load any data sources. Please check the data files.")
        st.stop()
    
    df = pd.concat(all_data, ignore_index=True)
    st.write(f"Total records loaded: {len(df)}")
    
    # Process embeddings
    cached_embeddings = process_texts_for_embeddings(df['combined_text'].tolist(), api_key)
    
    # Create input layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        query = st.text_area(
            "Beskriv seminar-temaet:",
            height=100,
            placeholder="Eksempel: Et seminar om klimatilpasning og hetebølger, med fokus på helsekonsekvenser for eldre."
        )
    if query:
        suggested_keywords = extract_relevant_keywords(query, CLIMATE_CATEGORIES)
            if suggested_keywords:
            st.divider()
            st.session_state.selected_keywords = render_keyword_selection(suggested_keywords)
                
                # Show selected keywords count
    if 'selected_keywords' not in st.session_state:
            st.session_state.selected_keywords = set()
            else:
                st.info("Ingen spesifikke nøkkelord funnet i beskrivelsen. Bruker generell semantisk matching.")
    
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
            value=0.35,  # Increased from 0.15
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
                # Use selected keywords in similarity calculation
                source_mask = df['source'].isin(selected_sources)
                filtered_df = df[source_mask].reset_index(drop=True)
                filtered_embeddings = [emb for emb, mask in zip(cached_embeddings, source_mask) if mask]
                
                # Pass selected keywords to find_similar_content
                results = find_similar_content(
                    query,
                    filtered_df,
                    filtered_embeddings,
                    api_key,
                    top_k=num_suggestions,
                    boost_keywords=selected_keywords  # New parameter
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
