import pandas as pd
import streamlit as st
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine
import os
from typing import Dict, List, Optional

# Define data source configurations as dictionaries
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
        "file_path": "data/stortinget-hearings.csv",  # Fixed filename to match actual file
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
        # Use specified separator if provided, otherwise default to comma
        separator = source_config.get("separator", ",")
        df = pd.read_csv(source_config["file_path"], sep=separator)
        
        # Remove any empty rows (as seen in the example)
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

def find_potential_speakers(query_text, df, embeddings, client, source_configs, top_k=5):
    """Find potential speakers based on content similarity across all sources"""
    query_embedding = get_embedding(query_text, client)
    if not query_embedding:
        return []
    
    # Calculate similarities
    similarities = []
    for emb in embeddings:
        similarity = 1 - cosine(query_embedding, emb)
        similarities.append(similarity)
    
    # Get top similar entries
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Extract speakers and their details
    speakers = []
    for idx in top_indices:
        entry = df.iloc[idx]
        source_config = next(s for s in source_configs if s["name"] == entry['source'])
        
        # Handle different speaker formats based on source
        if pd.notna(entry[source_config["speaker_column"]]):
            speaker_list = (entry[source_config["speaker_column"]].split('\n') 
                          if '\n' in str(entry[source_config["speaker_column"]]) 
                          else [entry[source_config["speaker_column"]]])
            
            for speaker in speaker_list:
                if speaker.strip():
                    speaker_info = {
                        'name': speaker.strip(),
                        'similarity': similarities[idx],
                        'source': source_config["name"],
                    }
                    
                    # Add optional fields if available
                    if source_config.get("event_column") and source_config["event_column"] in entry:
                        speaker_info['context'] = entry[source_config["event_column"]]
                    if source_config.get("date_column") and source_config["date_column"] in entry:
                        speaker_info['date'] = entry[source_config["date_column"]]
                    
                    speakers.append(speaker_info)
    
    # Remove duplicates while preserving highest similarity score
    seen = {}
    for speaker in speakers:
        if speaker['name'] not in seen or speaker['similarity'] > seen[speaker['name']]['similarity']:
            seen[speaker['name']] = speaker
    
    unique_speakers = list(seen.values())
    unique_speakers.sort(key=lambda x: x['similarity'], reverse=True)
    
    return unique_speakers

def main():
    st.set_page_config(
        page_title="Event Speaker Suggester",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Event Speaker Suggester")
    st.write("Enter your event topic to find relevant speaker suggestions from various sources.")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    # Load data from all sources
    all_data = []
    for source_config in DATA_SOURCES:
        df = load_source_data(source_config)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        st.error("No data sources could be loaded. Please check your data files and configurations.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    
    # Create two columns for input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_area(
            "Describe your event topic:",
            height=100,
            placeholder="Example: A seminar about Carbon Capture and Storage technology in Norway, focusing on policy frameworks and industry collaboration."
        )
    
    with col2:
        num_suggestions = st.slider(
            "Number of similar events to consider:",
            min_value=3,
            max_value=15,
            value=5
        )
        
        min_similarity = st.slider(
            "Minimum similarity score (0-1):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        # Add source filtering
        selected_sources = st.multiselect(
            "Filter by source:",
            options=[source["name"] for source in DATA_SOURCES],
            default=[source["name"] for source in DATA_SOURCES]
        )
    
    if st.button("Find Potential Speakers", type="primary"):
        if query:
            with st.spinner("Analyzing content..."):
                # Filter data by selected sources
                filtered_df = df[df['source'].isin(selected_sources)]
                
                # Get embeddings
                embeddings = [get_embedding(text, client) for text in filtered_df['combined_text']]
                
                suggestions = find_potential_speakers(
                    query, 
                    filtered_df, 
                    embeddings,
                    client,
                    DATA_SOURCES,
                    top_k=num_suggestions
                )
                
                # Filter by minimum similarity
                suggestions = [s for s in suggestions if s['similarity'] >= min_similarity]
                
                if suggestions:
                    st.subheader(f"Found {len(suggestions)} Potential Speakers")
                    
                    for i, speaker in enumerate(suggestions, 1):
                        with st.expander(f"🎤 {speaker['name']}", expanded=i<=3):
                            cols = st.columns([2, 1])
                            with cols[0]:
                                if 'context' in speaker:
                                    st.write("**Context:**", speaker['context'])
                                st.write("**Source:**", speaker['source'].title())
                            with cols[1]:
                                st.metric("Relevance Score", f"{speaker['similarity']:.2%}")
                                if 'date' in speaker:
                                    st.write("**Date:**", speaker['date'])
                    
                    st.download_button(
                        "Download Suggestions as CSV",
                        pd.DataFrame(suggestions).to_csv(index=False),
                        "speaker_suggestions.csv",
                        "text/csv",
                        key='download-csv'
                    )
                else:
                    st.warning("No speakers found matching your criteria. Try adjusting your filters or similarity threshold.")
                
                st.info("💡 These suggestions are based on speakers' past participation in various events and hearings. Consider factors like availability and current roles when making your final selection.")
        else:
            st.warning("Please enter a description of your event topic.")

if __name__ == "__main__":
    main()
