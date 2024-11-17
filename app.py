import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional, Set, Tuple
import time
import re

st.set_page_config(page_title="Seminar Deltaker Forslag", page_icon="🎯", layout="wide")

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
    },
    {
        "name": "Arendalsuka 2023",
        "file_path": "data/arendalsuka_2023_eventsognavn.csv",
        "text_columns": ['description', 'summary'],
        "speaker_column": 'person_names',
        "event_column": 'description',
        "content_column": 'summary',
        "separator": ","
    }
]

def extract_keywords_from_text(text: str) -> Set[str]:
    """Extract all meaningful words from text, excluding stop words"""
    words = re.findall(r'\w+', text.lower())
    keywords = {word for word in words 
               if word not in NORWEGIAN_STOP_WORDS 
               and len(word) > 2
               and not word.isdigit()}
    return keywords

@st.cache_data(show_spinner=True)
def load_source_data(source_config: Dict) -> Optional[pd.DataFrame]:
    """Load and prepare data from a single source"""
    try:
        st.write(f"Loading data from {source_config['name']}...")
        separator = source_config.get("separator", ",")
        df = pd.read_csv(source_config["file_path"], sep=separator)
        df = df.dropna(how='all')
        
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

def render_keyword_selection(keywords: Set[str], key_prefix: str = "") -> Set[str]:
    """Render interactive keyword selection interface"""
    selected_keywords = set()
    
    st.write("#### 🏷️ Velg relevante nøkkelord")
    st.write("Klikk for å velge/fjerne nøkkelord som er relevante for seminaret:")
    
    num_cols = 4
    cols = st.columns(num_cols)
    
    if 'keyword_states' not in st.session_state:
        st.session_state.keyword_states = {}
    
    st.markdown("""
        <style>
        .stButton > button {
            width: 100%;
            margin: 2px 0;
            padding: 2px 8px;
            font-size: 0.9em;
        }
        .keyword-selected {
            background-color: #e3f2fd;
            color: #1e88e5;
        }
        .keyword-deselected {
            background-color: #ffebee;
            color: #e53935;
        }
        </style>
    """, unsafe_allow_html=True)
    
    for idx, keyword in enumerate(sorted(keywords)):
        col_idx = idx % num_cols
        with cols[col_idx]:
            key = f"{key_prefix}_{keyword}"
            if key not in st.session_state.keyword_states:
                st.session_state.keyword_states[key] = True
            
            if st.button(
                "✓ " + keyword if st.session_state.keyword_states[key] else "○ " + keyword,
                key=key,
                help=f"Click to toggle '{keyword}'",
                use_container_width=True
            ):
                st.session_state.keyword_states[key] = not st.session_state.keyword_states[key]
                st.rerun()
            
            if st.session_state.keyword_states[key]:
                selected_keywords.add(keyword)
    
    return selected_keywords

def calculate_similarity(
    query_embedding: List[float],
    doc_embedding: List[float],
    query_text: str,
    doc_text: str,
    boost_keywords: Set[str] = None
) -> Tuple[float, Set[str]]:
    """Calculate similarity with better category matching"""
    if not query_embedding or not doc_embedding:
        return 0.0, set()
    
    # Base semantic similarity
    cos_sim = 1 - cosine(query_embedding, doc_embedding)
    
    # Text preprocessing
    query_lower = query_text.lower()
    doc_lower = doc_text.lower()
    
    # Find which categories the query matches
    matching_categories = set()
    for category, terms in CLIMATE_CATEGORIES.items():
        if any(term.lower() in query_lower for term in terms):
            matching_categories.add(category)
    
    # Calculate category matches in document
    category_matches = 0
    category_terms_found = set()
    for category in matching_categories:
        terms = CLIMATE_CATEGORIES[category]
        for term in terms:
            if term.lower() in doc_lower:
                category_matches += 1
                category_terms_found.add(term)
    
    # Word matching calculations
    query_words = {word for word in query_lower.split() 
                  if word not in NORWEGIAN_STOP_WORDS 
                  and len(word) > 2}
    
    doc_words = {word for word in doc_lower.split() 
                if word not in NORWEGIAN_STOP_WORDS 
                and len(word) > 2}
    
    # Find all matching words including category terms
    matching_words = query_words.intersection(doc_words).union(category_terms_found)
    
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
    
    # Category matching bonus
    category_bonus = category_matches / max(len(matching_categories), 1) if matching_categories else 0
    
    # Calculate final score with adjusted weights
    final_score = (
        0.25 * max(0, cos_sim) +                # Semantic similarity
        0.30 * category_bonus +                 # Increased weight for category matches
        0.25 * selected_keyword_ratio +         # Selected keywords
        0.20 * len(matching_words) / max(len(query_words), 1)  # Word matches
    )
    
    # Threshold adjustments
    if not matching_categories:
        final_score *= 0.5  # Penalize documents not matching any relevant categories
    
    if final_score < 0.3:
        final_score *= 0.5
    elif final_score > 0.8:
        final_score = 0.8 + (final_score - 0.8) * 0.5
    
    return final_score, matching_words

def highlight_text(text: str, keywords: Set[str]) -> str:
    """Highlight keywords in text using HTML with improved matching"""
    if not text or not keywords:
        return text
    
    # Prepare the text and keywords
    text_lower = text.lower()
    highlighted_text = text
    
    # Create a mapping of lowercase to original case keywords
    keyword_mapping = {}
    for keyword in keywords:
        # Find all occurrences in original text preserving case
        matches = re.finditer(re.escape(keyword.lower()), text_lower)
        for match in matches:
            start, end = match.span()
            original_case = text[start:end]
            keyword_mapping[original_case.lower()] = original_case
    
    # Sort keywords by length (longest first) to avoid partial matches
    sorted_keywords = sorted(keyword_mapping.keys(), key=len, reverse=True)
    
    # Highlight each keyword
    for keyword_lower in sorted_keywords:
        original_case = keyword_mapping[keyword_lower]
        pattern = re.compile(f'({re.escape(original_case)})', re.IGNORECASE)
        highlighted_text = pattern.sub(
            r'<span style="background-color: #fff3cd; padding: 0.1rem; border-radius: 0.2rem;">\1</span>',
            highlighted_text
        )
    
    return highlighted_text
    
    def find_similar_content(query_text: str, df: pd.DataFrame, cached_embeddings: List[List[float]], 
                        api_key: str, top_k: int = 5, boost_keywords: Set[str] = None) -> List[Dict]:
    """Find similar content with climate research context"""  # <-- Note the indentation here
    
    climate_context = "I kontekst av klimaforskning, energi, og bærekraftig omstilling: "
    enhanced_query = climate_context + query_text
    
    query_embedding = get_embedding(enhanced_query, api_key)
    if not query_embedding:
        return []
    
    similarities = []
    for i, emb in enumerate(cached_embeddings):
        if emb:
            similarity, matching_words = calculate_similarity(
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
                'matching_words': matching_words
            })
        else:
            similarities.append({
                'index': i,
                'score': 0,
                'source': df.iloc[i]['source'],
                'matching_words': set()
            })
    
    # Filter and sort results
    max_per_source = min(3, top_k)
    results_by_source = {}
    
    sorted_similarities = sorted(
        [s for s in similarities if s['score'] > 0.2],
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
                if isinstance(entry[source_config["speaker_column"]], str):
                    if '\n' in entry[source_config["speaker_column"]]:
                        speakers = [s.strip() for s in entry[source_config["speaker_column"]].split('\n')]
                    else:
                        speakers = [entry[source_config["speaker_column"]].strip()]
            
            results.append({
                'name': speakers[0] if speakers else "Unknown",
                'speakers': speakers,
                'similarity': float(sim['score']),
                'source': entry['source'],
                'context': entry.get(source_config["event_column"], ''),
                'content': entry.get(source_config["content_column"], ''),
                'matching_words': list(sim['matching_words'])
            })
    
    return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]

def highlight_text(text: str, keywords: Set[str]) -> str:
    """Highlight keywords in text using HTML"""
    if not text or not keywords:
        return text
    
    highlighted_text = text
    for keyword in sorted(keywords, key=len, reverse=True):
        pattern = re.compile(f'({re.escape(keyword)})', re.IGNORECASE)
        highlighted_text = pattern.sub(
            r'<span style="background-color: #fff3cd; padding: 0.1rem; border-radius: 0.2rem;">\1</span>',
            highlighted_text
        )
    
    return highlighted_text

def main():
    st.title("🎯 Seminar Deltaker Forslag")
    st.write("Beskriv seminaret ditt for å få forslag til relevante deltakere.")
    
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OpenAI API key not found in secrets!")
        st.stop()
    
    api_key = st.secrets["OPENAI_API_KEY"]
    
    all_data = []
    for source_config in DATA_SOURCES:
        source_df = load_source_data(source_config)
        if source_df is not None:
            all_data.append(source_df)
    
    if not all_data:
        st.error("Could not load any data sources.")
        st.stop()
    
    df = pd.concat(all_data, ignore_index=True)
    cached_embeddings = [get_embedding(text, api_key) for text in df['combined_text']]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_area(
            "Beskriv seminar-temaet:",
            height=100,
            placeholder="Eksempel: Et seminar om klimatilpasning og hetebølger, med fokus på helsekonsekvenser for eldre."
        )
        
        if query:
            extracted_keywords = extract_keywords_from_text(query)
            if extracted_keywords:
                selected_keywords = render_keyword_selection(extracted_keywords)
            else:
                st.info("Ingen nøkkelord funnet i beskrivelsen. Vennligst prøv å være mer spesifikk.")
    
    with col2:
        num_suggestions = st.slider("Antall forslag:", 3, 15, 5)
        min_similarity = st.slider("Minimum relevans:", 0.0, 1.0, 0.35, 0.05)
        
        selected_sources = st.multiselect(
            "Filtrer kilder:",
            options=[source["name"] for source in DATA_SOURCES],
            default=[source["name"] for source in DATA_SOURCES]
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
                    num_suggestions,
                    selected_keywords if 'selected_keywords' in locals() else None
                )
                
                if results:
                    st.subheader(f"🎯 Fant {len(results)} potensielle deltakere")
                    
                    for i, result in enumerate(results, 1):
                        if result['similarity'] >= min_similarity:
                            with st.expander(
                                f"🎤 {result['name']} - {result['similarity']:.1%} relevans", 
                                expanded=i<=3
                            ):
                                cols = st.columns([2, 1])
                                with cols[0]:
                                    st.write("**Kontekst:**", result['context'])
                                    if result['content']:
                                        st.write("**Beskrivelse:**")
                                        st.markdown(
                                            highlight_text(
                                                result['content'], 
                                                set(result['matching_words'])
                                            ), 
                                            unsafe_allow_html=True
                                        )
                                
                                with cols[1]:
                                    st.metric("Relevans", f"{result['similarity']:.1%}")
                                    st.write("**Kilde:**", result['source'])
                    
                    if results:
                        st.download_button(
                            "Last ned forslag som CSV",
                            pd.DataFrame(results).to_csv(index=False),
                            "deltaker_forslag.csv",
                            "text/csv",
                            key='download-csv'
                        )
                else:
                    st.warning("Ingen deltakere møtte minimumskravet for relevans.")
        else:
            st.warning("Vennligst beskriv seminar-temaet.")

if __name__ == "__main__":
    main()
