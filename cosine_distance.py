#  Copyright (c) 2020.  WeirdData
#  Author: Rohit Suratekar
#
#  Dataset used: MovieLens 20M
#  https://www.kaggle.com/grouplens/movielens-20m-dataset
#
#  Movie prediction based on cosine-distance

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from constants import *

FILE_RATINGS = "data/temp_rating.csv"  # Path to rating.csv
FILE_MOVIES = "data/movie.csv"  # Path to movie.csv
FILE_TAGS = "data/temp_tag.csv"  # Path to tag.cdv
MIN_VOTES = 500


def parse_ratings():
    # 'userId', 'movieId', 'rating', 'timestamp'
    df = pd.read_csv(FILE_RATINGS)
    return df


def get_movies():
    def _separate_year(m):
        _temp = m.rsplit("(", 1)
        if len(_temp) == 1:
            return None
        return _temp[-1].replace(")", "").strip()

    def _separate_title(m):
        _temp = m.rsplit("(", 1)
        if len(_temp) == 1:
            return m
        else:
            return _temp[0].strip()

    # 'movieId', 'title', 'genres'
    df = pd.read_csv(FILE_MOVIES)
    # Remove where Genres are not listed
    df = df[df[MOVIE_GENRES] != "(no genres listed)"]
    df[YEAR] = df[MOVIE_TITLE].map(lambda x: _separate_year(x))
    df[MOVIE_TITLE] = df[MOVIE_TITLE].map(lambda x: _separate_title(x))
    return df.reset_index(drop=True)


def parse_tags():
    """
    Collapses all tags to generate the tag string which can be directly
    used in calculating distance
    """

    def _collapse_tags(m):
        _all_tags = [str(x) for x in m]
        _all_tags = list(set(_all_tags))
        return " ".join(_all_tags)

    df = pd.read_csv(FILE_TAGS)
    df = df[[TAG, MOVIE_ID]]
    df = df.groupby(by=MOVIE_ID).agg(list)
    df[TAG] = df[TAG].apply(lambda x: _collapse_tags(x))
    df = df.reset_index()
    return df


def add_tags(movies: pd.DataFrame):
    """
    Adds tags to the existing movies
    :param movies: Movie Dataframe
    """
    tags = parse_tags().set_index(MOVIE_ID)
    movies = movies.set_index(MOVIE_ID)
    tags = pd.concat([tags, movies], join="inner", axis=1).reset_index()
    tags = tags.sort_values(by=MOVIE_ID)
    return tags


def add_ratings(movies: pd.DataFrame):
    """
    Normalizes ratings according to formula given in following and adds it
    to the dataframe
    https://stats.stackexchange.com/questions/6418/rating-system-taking-account-of-number-of-votes
    :param movies: Movie dataframe
    """
    ratings = parse_ratings()
    ratings = ratings.groupby(by=MOVIE_ID).agg(
        {USER_ID: "count", RATING: "sum"})
    ratings["avg"] = ratings[RATING] / ratings[USER_ID]
    mean_votes = ratings["avg"].mean()
    ratings["corr"] = (
            (ratings[USER_ID] / (ratings[USER_ID] + MIN_VOTES))
            * ratings["avg"] +
            (MIN_VOTES / (ratings[USER_ID] + MIN_VOTES)) * mean_votes)
    movies = movies.set_index(MOVIE_ID)
    del ratings[RATING]
    ratings[RATING] = ratings["corr"]
    ratings = ratings[[RATING]]
    ratings = pd.concat([ratings, movies], join="inner", axis=1).reset_index()
    ratings = ratings.sort_values(by=MOVIE_ID)
    return ratings


def convert_to_distance(data: pd.DataFrame, column):
    """
    Generates the distance matrix for given column
    :param data: Movie Datafram with given column
    :param column: name of the column which will be used to generate the
    distance matrix
    """
    all_values = data[column].values
    cv1 = CountVectorizer()
    k1 = cv1.fit_transform(all_values)
    dm1 = cosine_similarity(k1.toarray())
    return dm1


def generate_score(mapping, tag_score, cat_score):
    """
    Generates final score
    :param mapping: Movies dataframe
    :param tag_score: Row with desired movie from tag distance matrix
    :param cat_score: Row with desired movie from genre distance matrix
    """
    df = pd.DataFrame(data={MOVIE_TITLE: mapping[MOVIE_TITLE].values,
                            TAG: tag_score, MOVIE_GENRES: cat_score,
                            RATING: mapping[RATING].values})
    # Final Score
    # Movie category or genre will be wighted to be half
    df["total"] = df[TAG] + df[MOVIE_GENRES] * 0.5
    # Sort based on the final score
    df = df.sort_values(by=["total", RATING], ascending=False).reset_index(
        drop=True)
    print(df.head(10))


def run():
    # Your movie name
    desired_movie = "Jumanji"

    print(f"Analysis started for '{desired_movie}'")
    movies = get_movies()
    tags = add_tags(movies)
    ratings = add_ratings(tags)
    tag_dm = convert_to_distance(ratings, TAG)
    cat_dm = convert_to_distance(ratings, MOVIE_GENRES)
    ind = list(ratings[MOVIE_TITLE].values).index(desired_movie)
    generate_score(ratings, tag_dm[ind], cat_dm[ind])
