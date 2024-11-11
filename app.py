import pandas as pd
import streamlit as st
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine
import os

class DataSource:
    def __init__(self, name, file_path, text_columns, speaker_column, date_column=None, event_column=None):
        """
        Initialize a data source with its specific schema
        
        Parameters:
        - name: identifier for the data source (e.g., 'arendalsuka', 'parliament')
        - file_path: path to the CSV file
        - text_columns: list of column names to combine for similarity matching
        - speaker_column: column name containing speaker information
        - date_column: optional column name for event/hearing date
        - event_column: optional column name for event/hearing title
        """
        self.name = name
        self.file_path = file_path
        self.text_columns = text_columns
        self.speaker_column = speaker_column
        self.date_column = date_column
        self.event_column = event_column
        
    def load_and_prepare(self):
        """Load and prepare data from this source"""
        df = pd.read_csv(self.file_path)
        
        # Combine specified text columns for embedding
        df['combined_text'] = ''
        for col in self.text_columns:
            if col in df.columns:
                df['combined_text'] += ' ' + df[col].fillna('')
        
        df['combined_text'] = df['combined_text'].str.strip()
        df['source'] = self.name
        
        return df

@st.cache_data
def load_all_data(data_sources):
    """Load and combine data from all sources"""
    all_data = []
    for source in data_sources:
        try:
            df = source.load_and_prepare()
            all_data.append(df)
        except Exception as e:
            st.error(f"Error loading data from {source.name}: {str(e)}")
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

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

def find_potential_speakers(query_text, df, embeddings, client, data_sources, top_k=5):
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
        source = next(s for s in data_sources if s.name == entry['source'])
        
        # Handle different speaker formats based on source
        if pd.notna(entry[source.speaker_column]):
            speaker_list = entry[source.speaker_column].split('\n') if '\n' in str(entry[source.speaker_column]) else [entry[source.speaker_column]]
            
            for speaker in speaker_list:
                if speaker.strip():
                    speaker_info = {
                        'name': speaker.strip(),
                        'similarity': similarities[idx],
                        'source': source.name,
                    }
                    
                    # Add optional fields if available
                    if source.event_column and source.event_column in entry:
                        speaker_info['context'] = entry[source.event_column]
                    if source.date_column and source.date_column in entry:
                        speaker_info['date'] = entry[source.date_column]
                    
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
    
    # Define data sources
    data_sources = [
        DataSource(
            name="arendalsuka",
            file_path="data/arendalsuka_events.csv",
            text_columns=['title', 'header', 'om_arrangementet'],
            speaker_column='medvirkende',
            date_column='date',
            event_column='title'
        ),
        # Example additional source - adjust according to your actual data
        DataSource(
            name="parliament",
            file_path="data/parliament_hearings.csv",
            text_columns=['hearing_title', 'description'],
            speaker_column='participants',
            date_column='hearing_date',
            event_column='hearing_title'
        )
    ]
    
    # Initialize OpenAI client
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    # Load all data
    with st.spinner("Loading data from all sources..."):
        df = load_all_data(data_sources)
    
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
            options=[source.name for source in data_sources],
            default=[source.name for source in data_sources]
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
                    data_sources,
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
