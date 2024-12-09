import json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import Levenshtein

def clean_game_name(name: str) -> str:
    """Remove trademark symbols and normalize game name."""
    cleaned = name.replace('®', '').replace('™', '').replace('©', '')
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()

class EnhancedVectorSpaceModel:
    def __init__(self, genre_weight: float = 0.7, category_weight: float = 0.3):
        self.genre_vocabulary = set()
        self.category_vocabulary = set()
        self.genre_vectors = {}
        self.category_vectors = {}
        self.genre_weight = genre_weight
        self.category_weight = category_weight
        self.total_games = 0
        self.game_lookup = {}
        
    def fit(self, games_data: List[Dict[str, str]]):
        """Build the vector space model from games data."""
        # First pass: build vocabularies and frequencies
        genre_doc_freq = defaultdict(int)
        category_doc_freq = defaultdict(int)
        
        for game in games_data:
            app_id = game['app_id']
            genres = game['genres'].split(',') if game['genres'] else []
            categories = game['categories'].split(',') if game['categories'] else []
            
            self.game_lookup[app_id] = {
                'genres': genres,
                'categories': categories
            }
            
            self.genre_vocabulary.update(genres)
            self.category_vocabulary.update(categories)
            
            for genre in set(genres):
                genre_doc_freq[genre] += 1
            for category in set(categories):
                category_doc_freq[category] += 1
        
        self.total_games = len(games_data)
        
        # Calculate IDF scores
        self.genre_idf = {
            genre: np.log(self.total_games / (genre_doc_freq[genre] + 1))
            for genre in self.genre_vocabulary
        }
        
        self.category_idf = {
            category: np.log(self.total_games / (category_doc_freq[category] + 1))
            for category in self.category_vocabulary
        }
        
        # Create vectors
        genre_to_idx = {genre: idx for idx, genre in enumerate(self.genre_vocabulary)}
        category_to_idx = {category: idx for idx, category in enumerate(self.category_vocabulary)}
        
        for game in games_data:
            app_id = game['app_id']
            genres = game['genres'].split(',') if game['genres'] else []
            categories = game['categories'].split(',') if game['categories'] else []
            
            # Create genre vector
            genre_vector = np.zeros(len(self.genre_vocabulary))
            for genre in genres:
                idx = genre_to_idx[genre]
                genre_vector[idx] = self.genre_idf[genre]
            
            # Normalize genre vector
            norm = np.linalg.norm(genre_vector)
            if norm > 0:
                genre_vector = genre_vector / norm
            
            # Create category vector
            category_vector = np.zeros(len(self.category_vocabulary))
            for category in categories:
                idx = category_to_idx[category]
                category_vector[idx] = self.category_idf[category]
            
            # Normalize category vector
            norm = np.linalg.norm(category_vector)
            if norm > 0:
                category_vector = category_vector / norm
            
            self.genre_vectors[app_id] = genre_vector
            self.category_vectors[app_id] = category_vector
    
    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def get_similar_games(self, query_game_id: str, top_k: int = 10) -> List[Tuple[str, float, float, float]]:
        """Find similar games based on genre and category vectors."""
        if query_game_id not in self.genre_vectors:
            return []
        
        query_genre_vector = self.genre_vectors[query_game_id]
        query_category_vector = self.category_vectors[query_game_id]
        similarities = []
        
        for game_id in self.genre_vectors.keys():
            if game_id != query_game_id:
                genre_sim = self.calculate_similarity(query_genre_vector, self.genre_vectors[game_id])
                category_sim = self.calculate_similarity(query_category_vector, self.category_vectors[game_id])
                
                genre_sim = max(0.0, min(1.0, genre_sim))
                category_sim = max(0.0, min(1.0, category_sim))
                
                combined_sim = (self.genre_weight * genre_sim + self.category_weight * category_sim)
                similarities.append((game_id, combined_sim, genre_sim, category_sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

def get_closest_game_name(query: str, game_info_lookup: Dict, threshold: float = 0.6) -> str:
    """Find the most similar game name using fuzzy matching."""
    query = clean_game_name(query.lower().strip())
    best_ratio = 0
    best_match = None
    
    for game_info in game_info_lookup.values():
        game_name = clean_game_name(game_info['name'].lower())
        ratio = Levenshtein.ratio(query, game_name)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = game_info['name']
    
    if best_ratio >= threshold:
        return best_match
    return None

def load_and_process_games(json_file_path: str) -> tuple:
    """Load and process games from JSON file."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_games = []
    game_info_lookup = {}
    
    for app_id, game in data.items():
        genres = ','.join(game['genres']) if game['genres'] else ''
        categories = ','.join(game['categories']) if game['categories'] else ''
        
        processed_game = {
            'app_id': app_id,
            'genres': genres,
            'categories': categories
        }
        processed_games.append(processed_game)
        
        game_info_lookup[app_id] = {
            'name': game['name'],
            'genres': genres,
            'categories': categories,
            'price': game['price'],
            'release_date': game['release_date']
        }
    
    return processed_games, game_info_lookup