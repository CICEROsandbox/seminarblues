import pandas as pd
import streamlit as st
from openai import OpenAI
import numpy as np
from scipy.spatial.distances import cosine
import os

# Load and prepare the data
@st.cache_data
def load_data():
    # Update this path to where your CSV file is stored
    df = pd.read_csv('arendalsuka_events.csv')
    # Clean up and combine relevant text fields
    df['combined_text'] = df['title'] + ' ' + df['header'].fillna('') + ' ' + df['om_arrangementet'].fillna('')
    df['combined_text'] = df['combined_text'].fillna('')
    return df

# Get embeddings for a text using OpenAI
def get_embedding(text, client):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return None

# Find similar events and extract speakers
def find_similar_speakers(query_text, df, embeddings, client, top_k=5):
    # Get embedding for the query
    query_embedding = get_embedding(query_text, client)
    if not query_embedding:
        return []
    
    # Calculate similarities
    similarities = []
    for emb in embeddings:
        similarity = 1 - cosine(query_embedding, emb)
        similarities.append(similarity)
    
    # Get top similar events
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Extract speakers and their details
    speakers = []
    for idx in top_indices:
        event = df.iloc[idx]
        if pd.notna(event['medvirkende']):
            # Split speakers and process each one
            event_speakers = [s.strip() for s in event['medvirkende'].split('\n')]
            for speaker in event_speakers:
                if speaker:  # Only add non-empty speakers
                    speakers.append({
                        'name': speaker,
                        'similarity': similarities[idx],
                        'event': event['title'],
                        'event_description': event['om_arrangementet'] if pd.notna(event['om_arrangementet']) else '',
                        'date': event['date'] if pd.notna(event['date']) else ''
                    })
    
    # Remove duplicates while preserving highest similarity score
    seen = {}
    unique_speakers = []
    for speaker in speakers:
        if speaker['name'] not in seen or speaker['similarity'] > seen[speaker['name']]['similarity']:
            seen[speaker['name']] = speaker
    
    # Convert back to list and sort by similarity
    unique_speakers = list(seen.values())
    unique_speakers.sort(key=lambda x: x['similarity'], reverse=True)
    
    return unique_speakers

def main():
    st.set_page_config(
        page_title="CICERO Speaker Suggester",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 CICERO Speaker Suggester")
    st.write("Enter a description of your breakfast seminar to find relevant speaker suggestions from Arendalsuka events.")
    
# Initialize OpenAI client using secret
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    # Load data
   try:
        with st.spinner("Loading event data..."):
            df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Create two columns for input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query input
        query = st.text_area(
            "Describe your seminar topic:",
            height=100,
            placeholder="Example: A breakfast seminar about Carbon Capture and Storage technology in Norway, focusing on policy frameworks and industry collaboration."
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
    
    if st.button("Find Speakers", type="primary"):
        if query:
            # Calculate embeddings for all events
            with st.spinner("Analyzing events..."):
                embeddings = [get_embedding(text, client) for text in df['combined_text']]
                
                suggestions = find_similar_speakers(
                    query, 
                    df, 
                    embeddings,
                    client,
                    top_k=num_suggestions
                )
                
                # Filter by minimum similarity
                suggestions = [s for s in suggestions if s['similarity'] >= min_similarity]
                
                if suggestions:
                    st.subheader(f"Found {len(suggestions)} Suggested Speakers")
                    
                    for i, speaker in enumerate(suggestions, 1):
                        with st.expander(f"🎤 {speaker['name']}", expanded=i<=3):
                            cols = st.columns([2, 1])
                            with cols[0]:
                                st.write("**Found in event:**", speaker['event'])
                                if speaker['event_description']:
                                    st.write("**Event description:**", speaker['event_description'])
                            with cols[1]:
                                st.metric("Relevance Score", f"{speaker['similarity']:.2%}")
                                if speaker['date']:
                                    st.write("**Event date:**", speaker['date'])
                    
                    st.download_button(
                        "Download Suggestions as CSV",
                        pd.DataFrame(suggestions).to_csv(index=False),
                        "speaker_suggestions.csv",
                        "text/csv",
                        key='download-csv'
                    )
                else:
                    st.warning("No speakers found matching your criteria. Try lowering the minimum similarity score or increasing the number of events to consider.")
                
                st.info("💡 These suggestions are based on similar events from Arendalsuka. Consider factors like availability and current roles when making your final selection.")
        else:
            st.warning("Please enter a description of your seminar.")

if __name__ == "__main__":
    main()
