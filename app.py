import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional, Set, Tuple
import time
import re

st.set_page_config(page_title="Seminar Deltaker Forslag", page_icon="🎯", layout="wide")

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

def normalize_norwegian_word(word: str) -> str:
    """Normalize Norwegian word to its base form for matching"""
    suffixes = [
        'ene', 'er', 'et', 'en', 'a',  # Basic noun endings
        'ere', 'est', 'este',  # Adjective comparison forms
    ]
    
    word = word.lower()
    if len(word) > 4:
        for suffix in suffixes:
            if word.endswith(suffix):
                word = word[:-len(suffix)]
                break
    return word

def are_words_related(word1: str, word2: str) -> Tuple[bool, float]:
    """Check if two words are related in Norwegian and return similarity score"""
    norm1 = normalize_norwegian_word(word1)
    norm2 = normalize_norwegian_word(word2)
    
    if norm1 == norm2:
        return True, 1.0
    
    if norm1 in norm2 or norm2 in norm1:
        longer = max(len(norm1), len(norm2))
        shorter = min(len(norm1), len(norm2))
        score = shorter / longer
        return True, score
    
    common_prefixes = {
        'klima', 'miljø', 'natur', 'energi', 'kraft', 'samfunns',
        'bære', 'øko', 'sirkulær', 'fornybar', 'grønn', 'bio'
    }
    
    for prefix in common_prefixes:
        if (norm1.startswith(prefix) and norm2.startswith(prefix)) or \
           (norm1.endswith(prefix) and norm2.endswith(prefix)):
            return True, 0.8
    
    return False, 0.0
    
def highlight_keywords(text: str, keywords: Set[str]) -> str:
    """Highlight keywords in text with HTML styling"""
    if not text or not keywords:
        return text
    
    # Prepare text for highlighting
    highlighted_text = text
    
    # Create a list of all words to highlight, including normalized forms
    words_to_highlight = set()
    for keyword in keywords:
        # Handle the case where keyword contains an arrow (→) from our matching system
        base_keyword = keyword.split(' → ')[0] if ' → ' in keyword else keyword
        words_to_highlight.add(base_keyword)
        
        # Add normalized form
        norm_keyword = normalize_norwegian_word(base_keyword)
        words_to_highlight.add(norm_keyword)
        
        # Add common variations
        for suffix in ['er', 'ene', 'et', 'en', 'a']:
            words_to_highlight.add(norm_keyword + suffix)
    
    # Sort keywords by length (longest first) to avoid partial word highlighting
    sorted_keywords = sorted(words_to_highlight, key=len, reverse=True)
    
    # Create regex pattern for whole word matching with Norwegian characters
    pattern = r'\b(?:' + '|'.join(map(re.escape, sorted_keywords)) + r')\b'
    
    def replace_match(match):
        """Replace matched word with highlighted version"""
        word = match.group(0)
        return f'<span style="background-color: #faed27; padding: 0 2px; border-radius: 3px;">{word}</span>'
    
    # Replace case-insensitively but preserve original case
    highlighted_text = re.sub(pattern, replace_match, highlighted_text, flags=re.IGNORECASE)
    
    return highlighted_text

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

def extract_keywords_from_text(text: str) -> Set[str]:
    """Extract all meaningful words from text, excluding stop words"""
    # Clean text and split into words
    words = re.findall(r'\w+', text.lower())
    
    # Filter out stop words and short words
    keywords = {word for word in words 
               if word not in NORWEGIAN_STOP_WORDS 
               and len(word) > 2
               and not word.isdigit()}
    
    return keywords

def render_keyword_selection(keywords: Set[str], key_prefix: str = "") -> Set[str]:
    """Render interactive keyword selection interface"""
    selected_keywords = set()
    
    st.write("#### 🏷️ Velg relevante nøkkelord")
    st.write("Klikk for å velge/fjerne nøkkelord som er relevante for seminaret:")
    
    # Create columns for keyword display
    num_cols = 6  # Increase number of columns
    cols = st.columns(num_cols)
    
    # Create a dictionary to track button states
    if 'keyword_states' not in st.session_state:
        st.session_state.keyword_states = {}
    
    # Display keywords in columns
    for idx, keyword in enumerate(sorted(keywords)):
        col_idx = idx % num_cols
        with cols[col_idx]:
            # Initialize state for this keyword if not exists
            key = f"{key_prefix}_{keyword}"
            if key not in st.session_state.keyword_states:
                st.session_state.keyword_states[key] = True  # Default to selected
            
            # Create toggle button with custom styling
            if st.button(
                "✓ " + keyword if st.session_state.keyword_states[key] else "○ " + keyword,
                key=key,
                type="secondary",
                use_container_width=True,
                help=f"Click to toggle '{keyword}'",
                # Custom CSS for smaller, lighter buttons
                args=(),
                kwargs={'style': '''
                    <style>
                        div[data-testid="stButton"] button {
                            padding: 2px 8px;
                            font-size: 0.8em;
                            background-color: #f0f2f6;
                            border: none;
                            margin: 2px 0;
                        }
                        div[data-testid="stButton"] button:hover {
                            background-color: #e0e2e6;
                        }
                    </style>
                '''}
            ):
                # Toggle state when clicked
                st.session_state.keyword_states[key] = not st.session_state.keyword_states[key]
                st.rerun()
            
            # Add to selected keywords if state is True
            if st.session_state.keyword_states[key]:
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
    """Calculate semantic similarity with improved Norwegian keyword matching"""
    if not query_embedding or not doc_embedding:
        return 0.0, set()
    
    # Base similarity calculation
    cos_sim = 1 - cosine(query_embedding, doc_embedding)
    
    # Text preprocessing
    doc_words = {word for word in re.findall(r'\w+', doc_text.lower()) 
                if word not in NORWEGIAN_STOP_WORDS and len(word) > 2}
    
    # Enhanced keyword matching
    keyword_matches = []
    matched_keywords = set()
    
    if boost_keywords:
        for keyword in boost_keywords:
            best_match_score = 0
            best_match = None
            
            # Check each document word for a potential match
            for doc_word in doc_words:
                is_related, score = are_words_related(keyword, doc_word)
                if is_related and score > best_match_score:
                    best_match_score = score
                    best_match = doc_word
            
            if best_match_score > 0:
                keyword_matches.append((keyword, best_match, best_match_score))
                matched_keywords.add(f"{keyword} → {best_match}")
        
        # Calculate weighted keyword score
        if keyword_matches:
            keyword_score = sum(score for _, _, score in keyword_matches) / len(boost_keywords)
        else:
            keyword_score = 0
    else:
        keyword_score = 0
    
    # Calculate climate relevance
    climate_words = set()
    for word in doc_words:
        for climate_term in CLIMATE_KEYWORDS:
            is_related, score = are_words_related(word, climate_term)
            if is_related:
                climate_words.add(word)
                break
    
    climate_score = len(climate_words) / (len(doc_words) + 1)
    
    # Calculate final score
    final_score = (
        0.30 * max(0, cos_sim) +     # Semantic similarity
        0.40 * keyword_score +        # Keyword matching
        0.30 * climate_score         # Climate relevance
    )
    
    # Apply thresholds
    if keyword_score == 0 and boost_keywords:
        final_score *= 0.5
    elif final_score > 0.8:
        final_score = 0.8 + (final_score - 0.8) * 0.5
    
    return final_score, matched_keywords

def find_similar_content(query_text: str, df: pd.DataFrame, cached_embeddings: List[List[float]], 
                        api_key: str, top_k: int = 5, boost_keywords: Set[str] = None) -> List[Dict]:
    """Find similar content with emphasis on selected keywords"""
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
            similarity, matched_keywords = calculate_similarity(
                query_embedding, 
                emb,
                query_text,
                df.iloc[i]['combined_text'],
                boost_keywords
            )
            
            similarities.append({
                'index': i,
                'score': similarity,
                'source': df.iloc[i]['source'],
                'matched_keywords': matched_keywords  # Changed from matching_words
            })
            
            texts_for_debug.append((
                df.iloc[i]['combined_text'][:200],
                similarity,
                df.iloc[i]['source'],
                matched_keywords
            ))
        else:
            similarities.append({
                'index': i,
                'score': 0,
                'source': df.iloc[i]['source'],
                'matched_keywords': set()  # Changed from matching_words
            })
    
    # Enhanced debug information
    with st.expander("Search Analysis", expanded=False):
        st.write(f"Query: '{query_text}'")
        if boost_keywords:
            st.write(f"Selected keywords: {', '.join(boost_keywords)}")
        st.write("Top matches:")
        sorted_debug = sorted(texts_for_debug, key=lambda x: x[1], reverse=True)[:5]
        for text, sim, source, keywords in sorted_debug:
            st.write("\n---")
            st.write(f"Source: {source}")
            st.write(f"Score: {sim:.3f}")
            if keywords:
                st.write(f"Matched keywords: {', '.join(keywords)}")
            st.write(f"Text preview: {text}")
    
    # Filter and group results by source
    max_per_source = min(3, top_k)
    results_by_source = {}
    
    sorted_similarities = sorted(
        [s for s in similarities if isinstance(s, dict) and s['score'] > 0.2],
        key=lambda x: x['score'],
        reverse=True
    )
    
    for sim in sorted_similarities:
        source = sim['source']
        if source not in results_by_source:
            results_by_source[source] = []
        if len(results_by_source[source]) < max_per_source:
            results_by_source[source].append(sim)
    
    results = []
    for source_results in results_by_source.values():
        for sim in source_results:
            idx = sim['index']
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
                'similarity': float(sim['score']),
                'source': entry['source'],
                'context': entry.get(source_config["event_column"], ''),
                'content': entry.get(source_config["content_column"], ''),
                'matched_keywords': list(sim['matched_keywords'])  # Changed from matching_words
            })
    
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
    st.title("🎯 Seminar Deltaker Forslag")
    st.write("Beskriv seminaret ditt for å få forslag til relevante deltakere.")
    
    # Move initial information to expander
    with st.expander("ℹ️ Tips og informasjon", expanded=False):
        st.info("""
        💡 Tips for bedre resultater:
        - Vær spesifikk om temaet
        - Inkluder nøkkelord og konsepter
        - Beskriv formålet med seminaret
        """)
        
        # Add cache clear button here
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.rerun()
    
    # Check for API key
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OpenAI API key not found in secrets!")
        st.stop()
    
    api_key = st.secrets["OPENAI_API_KEY"]
    
    # Load data with status in expander
    with st.expander("📊 Data status", expanded=False):
        all_data = []
        for source_config in DATA_SOURCES:
            source_df = load_source_data(source_config)
            if source_df is not None:
                all_data.append(source_df)
        
        if not all_data:
            st.error("Could not load any data sources. Please check the data files.")
            st.stop()
        
        df = pd.concat(all_data, ignore_index=True)
        st.write(f"Total records loaded: {len(df)}")
        
        # Process embeddings
        cached_embeddings = process_texts_for_embeddings(df['combined_text'].tolist(), api_key)
    
    # Create input layout
    col1, col2, col3 = st.columns([2, 1, 1])
    selected_keywords = set()

    with col1:
        query = st.text_area(
            "Beskriv seminar-temaet:",
            height=100,
            placeholder="Eksempel: Et seminar om klimatilpasning og hetebølger, med fokus på helsekonsekvenser for eldre."
        )
        
        if query:
            # Extract all meaningful keywords from the query
            extracted_keywords = extract_keywords_from_text(query)
            
            if extracted_keywords:
                st.divider()
                selected_keywords = render_keyword_selection(extracted_keywords)
                
                if selected_keywords:
                    st.caption(f"Valgte nøkkelord: {len(selected_keywords)}")
            else:
                st.info("Ingen nøkkelord funnet i beskrivelsen. Vennligst prøv å være mer spesifikk.")

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
            value=0.35,
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
                # Now df is defined and we can use it safely
                source_mask = df['source'].isin(selected_sources)
                filtered_df = df[source_mask].reset_index(drop=True)
                filtered_embeddings = [emb for emb, mask in zip(cached_embeddings, source_mask) if mask]
                
                results = find_similar_content(
                    query,
                    filtered_df,
                    filtered_embeddings,
                    api_key,
                    top_k=num_suggestions,
                    boost_keywords=selected_keywords
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
                                    'source': result['source'],
                                    'matched_keywords': result['matched_keywords']  # Make sure this is included
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
                                cols = st.columns([2, 1])with cols[0]:
                                    if speaker['source'] == 'arendalsuka':
                                        st.markdown("**Deltaker i arrangement:** " + 
                                                highlight_keywords(speaker['context'], 
                                                                set(speaker['matched_keywords'])),
                                                unsafe_allow_html=True)
                                        if pd.notna(speaker['content']):
                                            st.markdown("**Arrangementsbeskrivelse:**")
                                            st.markdown(
                                                f'<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px;">'
                                                f'{highlight_keywords(speaker["content"], set(speaker["matched_keywords"]))}'
                                                f'</div>', 
                                                unsafe_allow_html=True
                                            )
                                    else:  # parliament hearings
                                        st.markdown("**Innspill til høring:** " + 
                                                highlight_keywords(speaker['context'], 
                                                                set(speaker['matched_keywords'])),
                                                unsafe_allow_html=True)
                                        if pd.notna(speaker['content']):
                                            st.markdown("**Høringsinnspill:**")
                                            st.markdown(
                                                f'<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px;">'
                                                f'{highlight_keywords(speaker["content"], set(speaker["matched_keywords"]))}'
                                                f'</div>', 
                                                unsafe_allow_html=True
                                            )
                                    
                                    st.markdown("**Kilde:** " + 
                                            ("Arendalsuka" if speaker['source'] == "arendalsuka" 
                                            else "Stortingshøringer"))
                                with cols[1]:
                                    st.metric("Relevans", f"{speaker['similarity']:.1%}")
                                    # Show matched keywords
                                    if speaker['matched_keywords']:
                                        st.write("**Matchende nøkkelord:**")
                                        for keyword in speaker['matched_keywords']:
                                            st.markdown(f"- {keyword}")
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

if __name__ == "__main__":
    main()
