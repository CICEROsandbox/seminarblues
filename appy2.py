from anthropic import Anthropic
import numpy as np
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional, Set, Tuple
import json
import time

def get_embedding(_text: str, _api_key: str) -> Optional[List[float]]:
    """Get embedding for a single text using Claude with optimized prompting"""
    try:
        client = Anthropic(api_key=_api_key)
        
        # Enhanced system prompt for more consistent embeddings
        system_prompt = """You are a semantic embedding system. Generate a dense vector representation that captures the semantic meaning of the input text, optimized for Norwegian language content and climate/energy domain knowledge. 
        
        Requirements:
        - Output only a JSON array of 1536 floating point numbers
        - Ensure vectors are L2 normalized
        - Emphasize domain-specific terminology in the embedding
        - Maintain consistent vector space for similar concepts in Norwegian and English
        """
        
        # Preprocess text for better embedding quality
        processed_text = _text.replace('\n', ' ').strip()
        if len(processed_text) > 1000:
            processed_text = processed_text[:1000] + "..."
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0,  # Use 0 temperature for consistent embeddings
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": f"Text to embed: {processed_text}"
            }]
        )
        
        try:
            embedding_vector = json.loads(response.content[0].text)
            if isinstance(embedding_vector, list) and len(embedding_vector) == 1536:
                # Normalize the vector
                norm = np.linalg.norm(embedding_vector)
                if norm > 0:
                    embedding_vector = [x / norm for x in embedding_vector]
                return embedding_vector
            else:
                raise ValueError("Invalid embedding format")
        except json.JSONDecodeError:
            st.error("Failed to parse embedding response")
            return None
            
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
    """Calculate semantic similarity optimized for Claude embeddings"""
    if not query_embedding or not doc_embedding:
        return 0.0, set()
    
    # Enhanced similarity calculation with L2 normalization
    query_norm = np.array(query_embedding)
    doc_norm = np.array(doc_embedding)
    
    # Calculate cosine similarity with normalized vectors
    cos_sim = np.dot(query_norm, doc_norm)
    
    # Text preprocessing with Norwegian-specific handling
    doc_words = {normalize_norwegian_word(word) for word in re.findall(r'\w+', doc_text.lower()) 
                if word not in NORWEGIAN_STOP_WORDS and len(word) > 2}
    
    # Enhanced keyword matching with context awareness
    keyword_matches = []
    matched_keywords = set()
    
    if boost_keywords:
        for keyword in boost_keywords:
            best_match_score = 0
            best_match = None
            
            # Check each document word for a potential match with context
            for doc_word in doc_words:
                is_related, score = are_words_related(keyword, doc_word)
                if is_related:
                    # Check surrounding words for context
                    context_score = calculate_context_relevance(keyword, doc_word, doc_text)
                    score = 0.7 * score + 0.3 * context_score
                    
                    if score > best_match_score:
                        best_match_score = score
                        best_match = doc_word
            
            if best_match_score > 0:
                keyword_matches.append((keyword, best_match, best_match_score))
                matched_keywords.add(f"{keyword} → {best_match}")
    
    # Calculate keyword score with position-aware weighting
    if keyword_matches:
        keyword_scores = []
        for keyword, match, score in keyword_matches:
            position_weight = calculate_position_weight(match, doc_text)
            weighted_score = score * position_weight
            keyword_scores.append(weighted_score)
        keyword_score = sum(keyword_scores) / len(boost_keywords)
    else:
        keyword_score = 0
    
    # Calculate climate relevance with domain-specific boosting
    climate_score = calculate_climate_relevance(doc_words, doc_text)
    
    # Adaptive scoring based on document length and quality
    doc_quality_factor = calculate_doc_quality(doc_text)
    
    # Calculate final score with adaptive weighting
    weights = calculate_adaptive_weights(cos_sim, keyword_score, climate_score)
    final_score = (
        weights['semantic'] * cos_sim +
        weights['keyword'] * keyword_score +
        weights['climate'] * climate_score
    ) * doc_quality_factor
    
    # Apply sigmoid normalization for more intuitive scoring
    final_score = 1 / (1 + np.exp(-5 * (final_score - 0.5)))
    
    return final_score, matched_keywords

def calculate_context_relevance(keyword: str, match: str, text: str) -> float:
    """Calculate contextual relevance of a keyword match"""
    # Get surrounding words
    words = text.lower().split()
    try:
        idx = words.index(match)
        context = words[max(0, idx-3):min(len(words), idx+4)]
        
        # Check for related terms in context
        context_score = 0
        for word in context:
            if any(are_words_related(word, term)[0] for term in CLIMATE_KEYWORDS):
                context_score += 0.2
        return min(1.0, context_score)
    except ValueError:
        return 0.0

def calculate_position_weight(word: str, text: str) -> float:
    """Calculate position-based weight for keyword matches"""
    words = text.lower().split()
    try:
        position = words.index(word)
        # Give higher weight to matches in first third of text
        relative_pos = position / len(words)
        if relative_pos <= 0.33:
            return 1.0
        elif relative_pos <= 0.66:
            return 0.8
        else:
            return 0.6
    except ValueError:
        return 0.5

def calculate_climate_relevance(words: Set[str], text: str) -> float:
    """Calculate climate relevance score with category-based weighting"""
    category_scores = {category: 0 for category in CLIMATE_CATEGORIES}
    
    for word in words:
        for category, terms in CLIMATE_CATEGORIES.items():
            for term in terms:
                is_related, score = are_words_related(word, term)
                if is_related:
                    category_scores[category] = max(category_scores[category], score)
    
    # Weight categories differently
    weighted_scores = {
        'Klimaendringer': 1.2,
        'Energi': 1.1,
        'Utslipp': 1.1,
        'Politikk og Omstilling': 0.9,
        'Miljø og Ressurser': 0.9,
        'Tilpasning og Risiko': 1.0,
        'Samfunn og Helse': 0.8
    }
    
    total_score = sum(score * weighted_scores[cat] for cat, score in category_scores.items())
    return min(1.0, total_score / (3 * len(CLIMATE_CATEGORIES)))

def calculate_doc_quality(text: str) -> float:
    """Calculate document quality factor"""
    # Check length
    length_score = min(1.0, len(text.split()) / 200)
    
    # Check for structure indicators
    structure_score = 0.5
    if any(indicator in text.lower() for indicator in ['konklusjon', 'sammendrag', 'formål', 'bakgrunn']):
        structure_score += 0.25
    if text.count('.') > text.count('!') + text.count('?'):  # Prefer formal language
        structure_score += 0.25
    
    return (length_score + structure_score) / 2

def calculate_adaptive_weights(semantic_score: float, keyword_score: float, climate_score: float) -> Dict[str, float]:
    """Calculate adaptive weights based on score distribution"""
    scores = [semantic_score, keyword_score, climate_score]
    max_score = max(scores)
    
    if max_score == semantic_score:
        return {'semantic': 0.5, 'keyword': 0.3, 'climate': 0.2}
    elif max_score == keyword_score:
        return {'semantic': 0.3, 'keyword': 0.5, 'climate': 0.2}
    else:
        return {'semantic': 0.3, 'keyword': 0.2, 'climate': 0.5}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def process_texts_for_embeddings(texts: List[str], _api_key: str) -> List[Optional[List[float]]]:
    """Process texts for embeddings with improved batching and error handling"""
    embeddings = []
    total = len(texts)
    batch_size = 5  # Process in smaller batches
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]
        status_text.text(f"Processing documents {i+1} to {min(i+batch_size, total)} of {total}")
        
        for text in batch_texts:
            try:
                emb = get_embedding(text, _api_key)
                embeddings.append(emb)
                progress_bar.progress((len(embeddings)) / total)
            except Exception as e:
                st.error(f"Error processing document {len(embeddings)+1}: {str(e)}")
                embeddings.append(None)
            time.sleep(0.5)  # Rate limiting
    
    progress_bar.empty()
    status_text.empty()
    st.write(f"Successfully processed {sum(1 for e in embeddings if e is not None)} of {total} embeddings")
    
    return embeddings

# Update imports at top of file:
from anthropic import Anthropic
import numpy as np
from scipy.spatial.distance import cosine
import json
import time