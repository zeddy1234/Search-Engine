import json
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

class VectorSpaceModel:
    # [Previous VectorSpaceModel class code remains exactly the same]
    def __init__(self):
        self.genre_vocabulary = set()  # Unique genres across all games
        self.genre_idf = {}  # IDF scores for genres
        self.game_vectors = {}  # Game ID -> genre vector mapping
        self.total_games = 0
        
    def fit(self, games_data: List[Dict[str, str]]):
        """
        Build the vector space model from games data.
        
        Args:
            games_data: List of game dictionaries, each containing at least 'app_id' and 'genres'
        """
        # First pass: build vocabulary and document frequency
        genre_doc_freq = defaultdict(int)
        
        for game in games_data:
            app_id = game['app_id']
            genres = game['genres'].split(',') if game['genres'] else []
            
            # Update vocabulary
            self.genre_vocabulary.update(genres)
            
            # Update document frequency
            for genre in set(genres):  # Use set to count each genre once per game
                genre_doc_freq[genre] += 1
        
        self.total_games = len(games_data)
        
        # Calculate IDF for each genre
        for genre in self.genre_vocabulary:
            self.genre_idf[genre] = np.log(self.total_games / (genre_doc_freq[genre] + 1))
        
        # Second pass: create TF-IDF vectors for each game
        genre_to_idx = {genre: idx for idx, genre in enumerate(self.genre_vocabulary)}
        vector_size = len(self.genre_vocabulary)
        
        for game in games_data:
            app_id = game['app_id']
            genres = game['genres'].split(',') if game['genres'] else []
            
            # Create vector with TF-IDF weights
            vector = np.zeros(vector_size)
            
            # Calculate term frequency for each genre
            genre_tf = defaultdict(int)
            for genre in genres:
                genre_tf[genre] += 1
            
            # Convert to TF-IDF
            for genre, tf in genre_tf.items():
                idx = genre_to_idx[genre]
                vector[idx] = tf * self.genre_idf[genre]
            
            # Normalize vector
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
                
            self.game_vectors[app_id] = vector
    
    def get_similar_games(self, query_game_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find games most similar to a query game based on genre vectors.
        
        Args:
            query_game_id: ID of the game to find similar games for
            top_k: Number of similar games to return
            
        Returns:
            List of (game_id, similarity_score) tuples, sorted by descending similarity
        """
        if query_game_id not in self.game_vectors:
            return []
        
        query_vector = self.game_vectors[query_game_id]
        similarities = []
        
        for game_id, game_vector in self.game_vectors.items():
            if game_id != query_game_id:
                # Calculate cosine similarity
                similarity = np.dot(query_vector, game_vector)
                similarities.append((game_id, similarity))
        
        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

def load_and_process_games(json_file_path: str) -> tuple:
    """
    Load games from JSON file and prepare them for the vector space model.
    
    Args:
        json_file_path: Path to the JSON file containing game data
        
    Returns:
        Tuple of (processed_games_data, game_info_lookup)
    """
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process games data
    processed_games = []
    game_info_lookup = {}
    
    for app_id, game in data.items():
        # Extract and join genres
        genres = ','.join(game['genres']) if game['genres'] else ''
        
        # Create processed game entry
        processed_game = {
            'app_id': app_id,
            'genres': genres
        }
        processed_games.append(processed_game)
        
        # Store full game info for lookup
        game_info_lookup[app_id] = {
            'name': game['name'],
            'genres': genres,
            'price': game['price'],
            'release_date': game['release_date']
        }
    
    return processed_games, game_info_lookup

def main():
    # Path to your JSON file
    json_file_path = 'C:/Users/bhush/Desktop/Info Project/games.json'  # Update this to your JSON file path
    
    # Load and process the data
    processed_games, game_info_lookup = load_and_process_games(json_file_path)
    
    # Initialize and fit the vector space model
    vsm = VectorSpaceModel()
    vsm.fit(processed_games)
    
    # Example: Find similar games for a specific game
    def find_similar_games(game_name: str, top_k: int = 5):
        # Find the game ID by name
        game_id = None
        for app_id, info in game_info_lookup.items():
            if info['name'].lower() == game_name.lower():
                game_id = app_id
                break
        
        if game_id is None:
            print(f"Game '{game_name}' not found.")
            return
        
        # Get similar games
        similar_games = vsm.get_similar_games(game_id, top_k)
        
        # Print results
        print(f"\nGames similar to '{game_name}':")
        print("-" * 50)
        for game_id, similarity in similar_games:
            game = game_info_lookup[game_id]
            print(f"Name: {game['name']}")
            print(f"Genres: {game['genres']}")
            print(f"Similarity Score: {similarity:.3f}")
            print(f"Price: ${game['price']}")
            print(f"Release Date: {game['release_date']}")
            print("-" * 50)
    
    # Example usage
    while True:
        game_name = input("\nEnter a game name to find similar games (or 'quit' to exit): ")
        if game_name.lower() == 'quit':
            break
        find_similar_games(game_name)

if __name__ == "__main__":
    main()