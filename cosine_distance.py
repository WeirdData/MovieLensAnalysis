#  Copyright (c) 2020.  WeirdData
#  Author: Rohit Suratekar
#
#  Dataset used: MovieLens 20M
#  https://www.kaggle.com/grouplens/movielens-20m-dataset
#
#  Movie prediction based on cosine-distance

import os

import h5py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import get_movies
from constants import *

FILE_RATINGS = "data/temp_rating.csv"  # Path to rating.csv
FILE_MOVIES = "data/movie.csv"  # Path to movie.csv
FILE_TAGS = "data/temp_tag.csv"  # Path to tag.cdv
MIN_VOTES = 500


def parse_ratings():
    # 'userId', 'movieId', 'rating', 'timestamp'
    df = pd.read_csv(FILE_RATINGS)
    return df


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


def generate_data():
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
    np.savez_compressed("data/tag_matrix", tag_dm)
    np.savez_compressed("data/genre_matrix", cat_dm)
    ratings[RATING] = round(ratings[RATING], 4)
    ratings[[MOVIE_TITLE, RATING]].to_csv("data/names.csv", index=False)
    print("Analysis finished")


def save_to_hdf5():
    if not os.path.exists("data/tag_matrix.npz"):
        print("Generating data")
        generate_data()
    tag_dm = np.load("data/tag_matrix.npz")["arr_0"]
    cat_dm = np.load("data/genre_matrix.npz")["arr_0"]
    with h5py.File("data/data.hdf5", "w") as f:
        f.create_dataset("tags", data=tag_dm)
        f.create_dataset("cats", data=cat_dm)


def find_similar_name(name: str, df: pd.DataFrame):
    print(f"Searching for similar1 titles like '{name}'")
    all_titles = list(df[MOVIE_TITLE].values)
    all_titles.append(name)
    cv1 = CountVectorizer(analyzer="char")

    k1 = cv1.fit_transform(all_titles).toarray()
    similar = []
    for m in range(0, len(k1) - 1):
        dist = cosine_similarity([k1[-1], k1[m]])[0, 1]
        if dist > 0:
            similar.append((all_titles[m], dist))

    if len(similar) == 0:
        raise Exception(f"Failed to find similar1 title. Please check "
                        f"spelling mistakes")
    similar = sorted(similar, key=lambda x: x[1], reverse=True)
    print("Following similar1 title found in current dataset...")
    for m in [x[0] for x in similar[:10]]:
        print(f"--> {m}")


def run():
    if not os.path.exists("data/data.hdf5"):
        print("Saving to database")
        save_to_hdf5()

    desired_movie = "Spider-Man"
    df = pd.read_csv("data/names.csv")
    idx = df.index[df[MOVIE_TITLE] == desired_movie]
    if idx.size == 0:
        find_similar_name(desired_movie, df)
        raise KeyError(f"{desired_movie} not found in current dataset.")
    idx = idx[0]

    with h5py.File("data/data.hdf5", "r") as f:
        tag_dm = f["tags"][idx]
        cat_dm = f["cats"][idx]
        generate_score(df, tag_dm, cat_dm)
