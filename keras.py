#  Copyright (c) 2020.  WeirdData
#  Author: Rohit Suratekar
#
#  Dataset used: MovieLens 20M
#  https://www.kaggle.com/grouplens/movielens-20m-dataset
#
#  Movie prediction based on neural network

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SecretColors import Palette
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

from common import get_movies
from constants import *

FILE_RATINGS = "data/rating.csv"  # Path to rating.csv
FILE_MOVIES = "data/movie.csv"  # Path to movie.csv
FILE_TAGS = "data/tag.csv"  # Path to tag.cdv
FILE_MOVIES_LABEL = "data/movie_labels.csv"
FILE_USER_LABEL = "data/user_labels.csv"

MODEL_NAME = "model"
LAYER_MOVIE_EMBEDDING = "MovieEmbedding"
LAYER_USER_EMBEDDING = "UserEmbedding"
LAYER_MOVIE_INPUT = "MovieInput"
LAYER_USER_INPUT = "UserInput"
LAYER_RATINGS_OUTPUT = "RatingsOutput"
USER_LABEL = "userLabel"
MOVIE_LABEL = "movieLabel"


def get_model(no_movies: int, no_users: int) -> keras.Model:
    ml = layers.Input(shape=(1,), name=LAYER_MOVIE_INPUT)
    me = layers.Embedding(no_movies, 16, name=LAYER_MOVIE_EMBEDDING)(ml)
    mo = layers.Flatten()(me)

    ul = layers.Input(shape=(1,), name=LAYER_USER_INPUT)
    ue = layers.Embedding(no_users, 16, name=LAYER_USER_EMBEDDING)(ul)
    uo = layers.Flatten()(ue)

    model = layers.concatenate([mo, uo])
    model = layers.Dense(32, activation="relu", name="HiddenLayer")(model)
    model = layers.Dense(1, activation="relu",
                         name=LAYER_RATINGS_OUTPUT)(model)
    model = keras.Model([ml, ul], model)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.MeanSquaredError())
    keras.utils.plot_model(model, f"{MODEL_NAME}.png", show_shapes=True)
    return model


def save_labels(df: pd.DataFrame):
    users = df[[USER_LABEL, USER_ID]]
    users = users.drop_duplicates().sort_values(by=USER_LABEL)
    movies = df[[MOVIE_LABEL, MOVIE_ID]]
    movies = movies.drop_duplicates().sort_values(by=MOVIE_LABEL)
    users.to_csv(FILE_USER_LABEL, index=False)
    movies.to_csv(FILE_MOVIES_LABEL, index=False)


def data_process(save_data=False) -> pd.DataFrame:
    # Avoid using this for general purpose
    mov_labels = LabelEncoder()
    usr_labels = LabelEncoder()
    # Get ratings
    rating = pd.read_csv(FILE_RATINGS)
    # Only take desired columns
    rating[MOVIE_LABEL] = mov_labels.fit_transform(rating[MOVIE_ID])
    rating[USER_LABEL] = usr_labels.fit_transform(rating[USER_ID])
    if save_data:
        save_labels(rating)
    rating[RATING] = rating[RATING] / 5
    rating = rating[[MOVIE_LABEL, USER_LABEL, RATING]]
    return rating


def save_history(hist, name):
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    p = Palette()
    plt.plot(train_loss, color=p.red(), label='Train Loss')
    plt.plot(val_loss, color=p.blue(), label='Validation Loss')
    plt.title("Train and Validation Loss Curve")
    plt.legend()
    plt.savefig(name)


def train_model():
    data = data_process()
    train, test = train_test_split(data, shuffle=True, test_size=0.2)
    no_of_movies = data[MOVIE_LABEL].nunique()
    no_of_users = data[USER_LABEL].nunique()
    model = get_model(no_of_movies, no_of_users)
    hist = model.fit({LAYER_MOVIE_INPUT: train[MOVIE_LABEL].values,
                      LAYER_USER_INPUT: train[USER_LABEL].values},
                     {LAYER_RATINGS_OUTPUT: train[RATING].values},
                     batch_size=200,
                     epochs=5,
                     verbose=1,
                     validation_data=(
                         {
                             LAYER_MOVIE_INPUT: test[MOVIE_LABEL].values,
                             LAYER_USER_INPUT: test[USER_LABEL].values},
                         {LAYER_RATINGS_OUTPUT: test[RATING].values}))

    model.save(f"{MODEL_NAME}.h5")
    save_history(hist, "fitting.png")


def cosine_similarity(v1, v2):
    return round(np.dot(v1, v2) / (np.linalg.norm(v1) * (np.linalg.norm(v2))),
                 3)


def _normalize_row(row, last):
    tmp_movies = np.zeros(last)
    tmp_ratings = np.zeros(last)
    movies = row[MOVIE_LABEL]
    ratings = row[RATING]
    for m, r in zip(movies, ratings):
        tmp_movies[m] = 1
        tmp_ratings[m] = r
    tmp = np.hstack([tmp_movies, tmp_ratings])
    row["temp"] = tmp
    return row


def find_closest_user(movies, ratings):
    """
    Finds closest user based on the given movies and their ratings
    :param movies: List of movie Labels
    :param ratings: List of ratings (on scale of 0-5)
    :return : Label of closest user
    """
    data = data_process()
    last_movie = data[MOVIE_LABEL].nunique()

    tmp_movies = np.zeros(last_movie)
    tmp_rating = np.zeros(last_movie)
    for m, r in zip(movies, ratings):
        tmp_movies[m] = 1
        tmp_rating[m] = r / 5
    input_data = np.hstack([tmp_movies, tmp_rating])

    data = data.groupby(by=USER_LABEL).agg(list).reset_index()
    current_user = 0
    current_distance = 0
    for _, row in data.iterrows():
        temp_row = _normalize_row(row, last_movie)
        dist = cosine_similarity(input_data, temp_row["temp"])
        if dist > current_distance:
            current_distance = dist
            current_user = temp_row[USER_LABEL]

    del data
    return current_user


def convert_movie_names(all_movies, names):
    mv = get_movies()
    mv = dict(zip(mv[MOVIE_ID], mv[MOVIE_TITLE]))
    ml = dict(zip(mv.values(), mv.keys()))
    labels = dict(zip(all_movies[MOVIE_ID], all_movies[MOVIE_LABEL]))
    # Convert movie names to ids
    input_movies = [ml[x] for x in names]
    # Convert movie IDs to movie labels
    for x in input_movies:
        try:
            labels[x]
        except KeyError:
            raise KeyError(f"'{mv[x]}' is not available in current analysis. "
                           f"Please use another movie")
    del mv
    return [labels[x] for x in input_movies]


def analyse(movie_list, rating_list):
    """
    Makes movie recommendation based on provided movie and their rating
    """
    original_movies = movie_list
    input_ratings = rating_list

    all_movies = pd.read_csv(FILE_MOVIES_LABEL)

    input_movies = convert_movie_names(all_movies, original_movies)
    user_id = find_closest_user(input_movies, input_ratings)
    all_movies[USER_LABEL] = user_id

    model = keras.models.load_model(f"{MODEL_NAME}.h5")  # type:keras.Model
    predictions = model.predict(
        [all_movies[MOVIE_LABEL], all_movies[USER_LABEL]])

    all_movies[RATING] = predictions
    all_movies = all_movies.sort_values(by=RATING, ascending=False)
    all_movies = all_movies.head(10 + len(original_movies))
    rating_map = dict(zip(all_movies[MOVIE_ID], all_movies[RATING]))
    all_movies = all_movies[MOVIE_ID].values

    mv = get_movies()
    mv = mv[mv[MOVIE_ID].isin(all_movies)]
    mv = mv[~mv[MOVIE_TITLE].isin(original_movies)]
    mv[RATING] = mv[MOVIE_ID].map(lambda x: rating_map[x])
    ratings = mv[RATING].values
    mv = mv[MOVIE_TITLE].values
    print("Recommended Movies [score]")
    for i, m in enumerate(zip(mv, ratings)):
        print(f"({i + 1}) {m[0]} [{round(m[1], 3)}]")


def run():
    # train_model()
    movies = ["Jumanji", "Toy Story"]
    ratings = [4.5, 1]
    analyse(movies, ratings)
