import streamlit as st
from game_search import (
    EnhancedVectorSpaceModel, 
    load_and_process_games, 
    get_closest_game_name,
    clean_game_name
)

# Set page configuration
st.set_page_config(
    page_title="Steam Game Finder",
    page_icon="🎮",
    layout="wide"
)

# Initialize session state
if 'vsm' not in st.session_state:
    try:
        # Load data and initialize model
        json_file_path = 'games.json'  # Update this path to your JSON file location
        processed_games, game_info_lookup = load_and_process_games(json_file_path)
        vsm = EnhancedVectorSpaceModel()
        vsm.fit(processed_games)
        
        st.session_state['vsm'] = vsm
        st.session_state['game_info_lookup'] = game_info_lookup
    except Exception as e:
        st.error(f"Error loading game data: {str(e)}")
        st.stop()

# Main interface
st.title("🎮 Steam Game Finder")
st.write("Find similar games based on genres and categories!")

# Add filters in sidebar
st.sidebar.title("Filters")
price_filter = st.sidebar.slider("Max Price ($)", 0, 100, 100)
show_free = st.sidebar.checkbox("Show Free Games Only")

# Main search interface
search_query = st.text_input("Enter a game name:", key="search")

if search_query:
    # Clean the search query
    cleaned_query = clean_game_name(search_query)
    
    # Find exact match
    game_id = None
    for app_id, info in st.session_state.game_info_lookup.items():
        if clean_game_name(info['name'].lower()) == cleaned_query.lower():
            game_id = app_id
            break
    
    if game_id is None:
        # Try fuzzy matching
        suggestion = get_closest_game_name(cleaned_query, st.session_state.game_info_lookup)
        if suggestion:
            st.warning(f"Game not found. Did you mean '{suggestion}'?")
            if st.button("Yes, search for this game"):
                search_query = suggestion
                # Find the game ID for the suggestion
                for app_id, info in st.session_state.game_info_lookup.items():
                    if info['name'] == suggestion:
                        game_id = app_id
                        break
        else:
            st.error("No similar games found.")
    
    if game_id:
        # Get similar games
        similar_games = st.session_state.vsm.get_similar_games(game_id)
        
        # Display results
        st.subheader(f"Games similar to '{search_query}'")
        
        # Filter games based on price
        filtered_games = []
        for game_id, combined_sim, genre_sim, category_sim in similar_games:
            game = st.session_state.game_info_lookup[game_id]
            price = float(game['price'])
            
            if show_free and price > 0:
                continue
            if price > price_filter:
                continue
                
            filtered_games.append((game_id, combined_sim, genre_sim, category_sim))
        
        if not filtered_games:
            st.info("No games found matching the current filters.")
        
        for game_id, combined_sim, genre_sim, category_sim in filtered_games:
            game = st.session_state.game_info_lookup[game_id]
            
            # Create an expander for each game
            with st.expander(f"🎮 {game['name']} (Similarity: {combined_sim:.3f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Genres:**", game['genres'])
                    st.write("**Categories:**", game['categories'])
                
                with col2:
                    st.write("**Price:**", f"${game['price']}")
                    st.write("**Release Date:**", game['release_date'])
                
                # Show similarity scores with progress bars
                st.write("**Similarity Scores:**")
                st.write("Combined:")
                st.progress(float(combined_sim))
                st.write("Genre:")
                st.progress(float(genre_sim))
                st.write("Category:")
                st.progress(float(category_sim))

# Add information in sidebar
with st.sidebar:
    st.markdown("---")
    st.header("About")
    st.write("""
    This tool helps you discover games similar to your favorites on Steam.
    
    It uses:
    - Genre matching
    - Category matching
    - Smart name matching
    - Price filtering
    
    The similarity scores are calculated using a combination of genre (70%) 
    and category (30%) matching.
    """)