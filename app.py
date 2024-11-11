import pandas as pd
import streamlit as st
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional

# System instructions for GPT
SYSTEM_INSTRUCTIONS = """
Du er en erfaren rådgiver som skal hjelpe med å finne relevante deltakere til seminarer og arrangementer. 
Din oppgave er å analysere foreslåtte deltakere og:
1. Gruppere deltakerne etter type (f.eks. forskere, politikere, organisasjoner, næringsliv)
2. Forklare hvorfor de vil være relevante for seminaret
3. Foreslå en god sammensetning av panel eller deltakerliste
4. Identifisere eventuelle perspektiver eller grupper som mangler
5. Gi konkrete råd om hvem som bør prioriteres å invitere og hvorfor

Bruk en uformell, men profesjonell tone. Vær konkret i anbefalingene.
"""

# Configuration
DATA_SOURCES = [
    {
        "name": "arendalsuka",
        "file_path": "data/arendalsuka_events.csv",
        "text_columns": ['title', 'header', 'om_arrangementet'],
        "speaker_column": 'medvirkende',
        "date_column": 'date',
        "event_column": 'title'
    },
    {
        "name": "parliament_hearings",
        "file_path": "data/stortinget-hearings.csv",
        "text_columns": ['Høringssak', 'Innhold - høring'],
        "speaker_column": 'Innsender',
        "event_column": 'Høringssak',
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

def get_embedding(text, client):
    """Get embeddings for a text using OpenAI"""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return None

def find_similar_content(query_text, df, client, top_k=5):
    """Find similar content and extract speakers"""
    query_embedding = get_embedding(query_text, client)
    if not query_embedding:
        return []
    
    # Calculate similarities
    embeddings = [get_embedding(text, client) for text in df['combined_text']]
    similarities = []
    for emb in embeddings:
        if emb:  # Check if embedding was successfully created
            similarity = 1 - cosine(query_embedding, emb)
            similarities.append(similarity)
        else:
            similarities.append(0)  # No similarity if embedding failed
    
    # Get top similar entries
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
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
        
        # Add result
        results.append({
            'speakers': speakers,
            'similarity': similarities[idx],
            'source': entry['source'],
            'context': entry[source_config["event_column"]] if source_config["event_column"] in entry else '',
            'combined_text': entry['combined_text']
        })
    
    return results

def get_gpt_analysis(query: str, speakers: List[Dict], client: OpenAI) -> str:
    """Get GPT analysis of the suggestions based on the seminar topic"""
    try:
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

    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        return

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
                filtered_df = df[df['source'].isin(selected_sources)]
                results = find_similar_content(query, filtered_df, client, top_k=num_suggestions)
                
                if results:
                    # Process speakers
                    speakers_dict = {}
                    for result in results:
                        for speaker in result['speakers']:
                            if speaker not in speakers_dict or result['similarity'] > speakers_dict[speaker]['similarity']:
                                speakers_dict[speaker] = {
                                    'similarity': result['similarity'],
                                    'context': result['context'],
                                    'source': result['source']
                                }
                    
                    speakers = [
                        {'name': name, **info}
                        for name, info in speakers_dict.items()
                        if info['similarity'] >= min_similarity
                    ]
                    speakers.sort(key=lambda x: x['similarity'], reverse=True)
                    
                    if speakers:
                        # Get GPT analysis
                        with st.spinner("Analyserer forslag..."):
                            analysis = get_gpt_analysis(query, speakers, client)
                            if analysis:
                                st.subheader("💡 Analyse og anbefalinger")
                                st.write(analysis)
                                st.divider()
                        
                        # Display detailed results
                        st.subheader(f"🎯 Fant {len(speakers)} potensielle deltakere")
                        
                        for i, speaker in enumerate(speakers, 1):
                            with st.expander(
                                f"🎤 {speaker['name']} - {speaker['similarity']:.0%} relevans", 
                                expanded=i<=3
                            ):
                                cols = st.columns([2, 1])
                                with cols[0]:
                                    st.write("**Funnet i:**", speaker['context'])
                                    st.write("**Kilde:**", 
                                           "Arendalsuka" if speaker['source'] == "arendalsuka" 
                                           else "Stortingshøringer")
                                with cols[1]:
                                    st.metric("Relevans", f"{speaker['similarity']:.2%}")
                        
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
