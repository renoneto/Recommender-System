import pandas as pd

def weighted_rating(df):
    """
    Calculates the IMDB's Weighted Rating using the following formula:
        (v / (v+m) * R) + (m / (m+v) * C)

    where:
    - v is the number of votes for the movie;
    - m is the minimum votes required to be listed in the chart;
    - R is the average rating of the movie; And
    - C is the mean vote across the whole report
    """
    v = df['count_rating']
    m = df['minimum_no_of_ratings']
    R = df['avg_rating']
    C = df['overall_avg_rating']

    return (v / (v+m) * R) + (m / (m+v) * C)

def filtered_dataset(genre, ratings_movies_df):

    # Keep only the selected genre
    genre_df = ratings_movies_df[ratings_movies_df['genres'].str.contains(genre)]

    # Calculate the 95th quantile and the weighted rating
    minimum_no_of_ratings = genre_df['count_rating'].quantile(0.95)
    print(minimum_no_of_ratings)
    genre_df['minimum_no_of_ratings'] = minimum_no_of_ratings
    genre_df['weighted_rating'] = genre_df.apply(weighted_rating, axis=1)

    # Remove movies with not enough ratings
    genre_df = genre_df[genre_df['count_rating'] >= minimum_no_of_ratings]

    # Sorted it by weighted rating so we have the highest ratings on the top
    genre_df = genre_df.sort_values('weighted_rating', ascending=False)
    genre_df = genre_df.reset_index(drop=True)

    # Keep certain relevant columns
    genre_df = genre_df[['movieId', 'title',
                         'genres', 'count_rating',
                         'minimum_no_of_ratings', 'weighted_rating']]

    return genre_df
