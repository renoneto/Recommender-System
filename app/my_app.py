from functions import filtered_dataset

import streamlit as st
import pandas as pd
from surprise import Reader, Dataset

# Import required data
current_user_ratings = pd.read_csv('./data/user_movie_ratings.csv')
movie_ratings = pd.read_csv('./data/movies_by_rating.csv')
model = pd.read_pickle('svd.pkl')['algo']

# Set default values
n_recommendations = 5
n_of_movies_to_rate = 5
default_user_id = 999999

# Set page title
st.set_page_config(page_title="Movies Recommendation")
st.write('# What movies to watch next? :clapper:')
st.subheader("Select a genre, rate five movies and I'm going to tell you what to watch next :crystal_ball:")

# Ask first question
st.write('#### :one: Please select your favorite genre')
genres = ('-', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
		  'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western')
selected_genre = st.selectbox('Genres:', genres)

# If user hasn't chosen a genre, prompt the user to choose one
if selected_genre == '-':
	st.warning('Please choose a genre')
	st.stop()

# Show the selected genre
st.write('Selected genre:', selected_genre)

# Filter dataset given the selected genre
favorite_genre_movies = filtered_dataset(selected_genre, movie_ratings)

# Keep the highest rated movies by keeping the top 20, shuffling and choosing 5
favorite_genre_movies_filtered = favorite_genre_movies.iloc[:20].sample(frac=1, random_state=111)
favorite_genre_movies_filtered = favorite_genre_movies_filtered.iloc[:n_of_movies_to_rate]

# Grab Ratings - Don't let users move on if no ratings
st.write('#### :two: Please rate the following movies from 1 to 5')
ratings_options = ('-', "Haven't watched it yet.", '1', '2', '3', '4', '5')

# Store User Ratings on New Movies
input_ratings = []

# Create select box for each movie
for n in range(n_of_movies_to_rate):
	user_movie_rating = st.selectbox(favorite_genre_movies_filtered.iloc[n]['title'], ratings_options)
	input_ratings.append(user_movie_rating)

	# Ask the user to rate the movie
	if user_movie_rating == '-':
		st.warning('Please rate the movie')
		st.stop()

# Click to show recommendations
if st.button('Show recommendations!'):

	# Create empty list to store ratings with more info
	new_ratings = []

	for idx, movie_rating in enumerate(input_ratings):
		if movie_rating != "Haven't watched it yet.":
			new_ratings.append({'userId': default_user_id,
								'movieId': favorite_genre_movies_filtered.iloc[idx]['movieId'],
								'rating': movie_rating})

	# Append new ratings to existing dataset
	new_ratings_df = pd.DataFrame(new_ratings)
	updated_df = pd.concat([new_ratings_df, current_user_ratings])

	# Transform data set
	reader = Reader()
	new_data = Dataset.load_from_df(updated_df, reader)
	new_dataset = new_data.build_full_trainset()

	# Retrain the model
	model.fit(new_dataset)

	# Make predictions for the user
	predictions = []
	for movie_id in favorite_genre_movies['movieId'].to_list():
		predicted_score = model.predict(default_user_id, movie_id)[3]
		predictions.append((movie_id, predicted_score))

	# order the predictions from highest to lowest rated
	ranked_movies = pd.DataFrame(predictions, columns=['movieId', 'predicted_score'])
	ranked_movies = ranked_movies[~ranked_movies['movieId'].isin(new_ratings_df['movieId'])]
	ranked_movies = ranked_movies.sort_values('predicted_score', ascending=False).reset_index(drop=True)
	ranked_movies = pd.merge(ranked_movies, movie_ratings, on='movieId')
	ranked_movies = ranked_movies[ranked_movies['movieId'].isin(favorite_genre_movies['movieId'])]

	# Show the recommendations
	st.write('### Here are the recommendations')

	# If there aren't enough movies to recommend then show only what's in there
	if len(ranked_movies) < n_recommendations:
		n_recommendations = len(ranked_movies)

	# Show recommendations
	for row in range(n_recommendations):
		movie_id = ranked_movies.iloc[row]['movieId']
		recommended_title = movie_ratings[movie_ratings['movieId'] == movie_id]['title'].item()
		st.write(f'###### #{row+1} {recommended_title}')
