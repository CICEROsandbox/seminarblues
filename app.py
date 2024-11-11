import pandas as pd
import streamlit as st
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine
import os
from typing import Dict, List, Optional

# System instructions for the GPT context
SYSTEM_INSTRUCTIONS = """
Du er en erfaren rådgiver som skal hjelpe med å finne relevante deltakere til seminarer og arrangementer. 
Din oppgave er å:
1. Analysere seminar-temaet som blir foreslått
2. Identifisere relevante personer og organisasjoner fra de tilgjengelige kildene (Arendalsuka og Stortingshøringer)
3. Forklare hvorfor disse vil være relevante deltakere basert på deres tidligere engasjement
4. Vurdere balansen mellom ulike perspektiver og interesser
5. Gi konkrete anbefalinger om hvem som bør inviteres og hvorfor

Vær spesielt oppmerksom på:
- Aktualitet og relevans til tema
- Balanse mellom ulike synspunkter og interesser
- Kombinasjon av praktisk erfaring og faglig ekspertise
- Representasjon fra både organisasjoner og enkeltpersoner
"""

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

# Rest of the code remains the same until the find_potential_speakers function

def get_gpt_analysis(query: str, suggestions: List[Dict], client: OpenAI) -> str:
    """Get GPT analysis of the suggestions based on the seminar topic"""
    try:
        # Prepare the context
        context = f"""
Seminartema: {query}

Potensielle deltakere basert på tidligere engasjement:
"""
        for s in suggestions:
            context += f"\n- {s['name']} (fra {s['source']})"
            if 'context' in s:
                context += f"\n  Kontekst: {s['context']}"

        # Get GPT analysis
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": context}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting GPT analysis: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Seminar Deltaker Forslag",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Seminar Deltaker Forslag")
    st.write("Beskriv seminaret ditt for å få forslag til relevante deltakere basert på tidligere arrangementer og høringer.")
    
    # Rest of the main function remains the same until after getting suggestions
    
    if st.button("Finn potensielle deltakere", type="primary"):
        if query:
            with st.spinner("Analyserer innhold..."):
                # Previous suggestion code remains the same
                
                if suggestions:
                    # Get GPT analysis
                    with st.spinner("Analyserer forslag..."):
                        analysis = get_gpt_analysis(query, suggestions, client)
                        if analysis:
                            st.subheader("Analyse og anbefalinger")
                            st.write(analysis)
                    
                    # Show detailed suggestions
                    st.subheader(f"Fant {len(suggestions)} potensielle deltakere")
                    
                    # Rest of the display code remains the same

# Rest of the code remains the same
