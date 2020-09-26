#  Copyright (c) 2020.  WeirdData
#  Author: Rohit Suratekar
#
#  Data analysis using pairwise distance
#
#  Dataset used: MovieLens 20M
#  https://www.kaggle.com/grouplens/movielens-20m-dataset

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from constants import *

FILE_RATINGS = "data/temp_rating.csv"
FILE_MOVIES = "data/movie.csv"
FILE_TAGS = "data/temp_tag.csv"
MIN_VOTES = 100


def parse_ratings():
    # 'userId', 'movieId', 'rating', 'timestamp'
    df = pd.read_csv(FILE_RATINGS)
    return df


def parse_movies():
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
    mask = df[MOVIE_GENRES] == "(no genres listed)"
    df.loc[mask, MOVIE_GENRES] = "Unknown"
    df[MOVIE_GENRES] = (df[MOVIE_GENRES].str.split("|")
                        .apply(lambda x: [y.strip() for y in x]))
    df[YEAR] = df[MOVIE_TITLE].map(lambda x: _separate_year(x))
    df[MOVIE_TITLE] = df[MOVIE_TITLE].map(lambda x: _separate_title(x))
    return df


def parse_tags():
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


def get_tag_string(movies: pd.DataFrame, tags: pd.DataFrame, names: list):
    m = movies[movies[MOVIE_TITLE].isin(names)]
    m.set_index(MOVIE_ID)
    t = tags.set_index(MOVIE_ID)
    m = pd.concat([m, t], join='inner', axis=1)
    return " ".join(m[TAG].values)


def correct_ranking(movies: pd.DataFrame, ranking: pd.DataFrame, movie_list):
    new_list = []
    for m in movie_list:
        current_movie = movies[MOVIE_TITLE].values[m[0]]
        movie_id = movies[movies[MOVIE_TITLE] == current_movie][
            MOVIE_ID].values[0]
        rating = ranking[ranking[MOVIE_ID] == movie_id]["corr"].values[0]
        new_list.append((current_movie, m[1], rating))

    new_list = sorted(new_list, key=lambda x: x[1] * x[2], reverse=True)
    for m in new_list:
        print(f"{m[0]} [rating: {round(m[2], 2)}, similarity: "
              f"{round(m[1], 3)}] ")


def run():
    print("Analysis Started...")
    # Get tags and movies
    tags = parse_tags()
    movies = parse_movies()
    ratings = parse_ratings()
    ratings = ratings.groupby(by=MOVIE_ID).agg(
        {USER_ID: "count", RATING: "sum"})
    ratings["avg"] = ratings[RATING] / ratings[USER_ID]
    mean_votes = ratings["avg"].mean()
    ratings["corr"] = (
            (ratings[USER_ID] / (ratings[USER_ID] + MIN_VOTES))
            * ratings["avg"] +
            (MIN_VOTES / (ratings[USER_ID] + MIN_VOTES)) * mean_votes)
    ratings = ratings.reset_index()

    # Define our target
    interested_movies = ["Jumanji"]
    print(f"Interested Movies: {interested_movies}\n")
    target = get_tag_string(movies, tags, interested_movies)

    # Generate sparse matrix by counting the tags in specific locations
    cv = CountVectorizer()
    all_tags = tags[TAG].values
    all_tags = np.hstack((all_tags, target))
    cm = cv.fit_transform(all_tags)

    # Generate cosine distance of the data
    distance = cosine_similarity(cm.toarray())

    # Our target is last row
    target_dist = distance[-1]
    target_dist = list(enumerate(target_dist))
    target_dist = sorted(target_dist, key=lambda x: x[1], reverse=True)
    c = 0
    start = None
    all_rec = []
    for m in target_dist:
        current_movie = movies[MOVIE_TITLE].values[m[0]]
        if current_movie not in interested_movies:
            all_rec.append(m)
            if start is None:
                start = m[1]
            c += 1
            if c > 4 and m[1] != start:
                break

    correct_ranking(movies, ratings, all_rec)
