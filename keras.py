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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from constants import *
from scipy.sparse import csr_matrix

FILE_RATINGS = "data/temp_rating.csv"  # Path to rating.csv
FILE_MOVIES = "data/movie.csv"  # Path to movie.csv
FILE_TAGS = "data/temp_tag.csv"  # Path to tag.cdv
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

    model = layers.Dot(1)([mo, uo])
    model = layers.Dense(16, activation="relu", name="HiddenLayer")(model)
    model = layers.Dense(1, activation="relu",
                         name=LAYER_RATINGS_OUTPUT)(model)
    model = keras.Model([ml, ul], model)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.MeanSquaredError())
    keras.utils.plot_model(model, f"{MODEL_NAME}.png", show_shapes=True)
    return model


def get_user_model(no_of_movies, no_of_users):
    model = keras.Sequential(
        [
            layers.Input(shape=(no_of_movies, 2), name="InputLayer"),
            layers.Embedding(no_of_movies, 128, name="Embedded"),
            layers.Dense(128, activation="relu", name="HiddenLayer1"),
            layers.Dense(1, activation="softmax", name="OutputLayer")
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError()
    )
    keras.utils.plot_model(model, f"{MODEL_NAME}_sub.png", show_shapes=True)
    return model


def save_labels(df: pd.DataFrame):
    users = df[[USER_LABEL, USER_ID]]
    users = users.drop_duplicates().sort_values(by=USER_LABEL)
    movies = df[[MOVIE_LABEL, MOVIE_ID]]
    movies = movies.drop_duplicates().sort_values(by=MOVIE_LABEL)
    users.to_csv(FILE_USER_LABEL, index=False)
    movies.to_csv(FILE_MOVIES_LABEL, index=False)


def data_process() -> pd.DataFrame:
    mov_labels = LabelEncoder()
    usr_labels = LabelEncoder()
    # Get ratings
    rating = pd.read_csv(FILE_RATINGS)
    # Only take desired columns
    rating[MOVIE_LABEL] = mov_labels.fit_transform(rating[MOVIE_ID])
    rating[USER_LABEL] = usr_labels.fit_transform(rating[USER_ID])
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
                     batch_size=128,
                     epochs=5,
                     verbose=1,
                     validation_data=(
                         {
                             LAYER_MOVIE_INPUT: test[MOVIE_LABEL].values,
                             LAYER_USER_INPUT: test[USER_LABEL].values},
                         {LAYER_RATINGS_OUTPUT: test[RATING].values}))

    model.save(MODEL_NAME)
    save_history(hist, "fitting.png")


def cosine_similarity(v1, v2):
    return round(np.dot(v1, v2) / (np.linalg.norm(v1) * (np.linalg.norm(v2))),
                 3)


def train_similar_user_model():
    data = data_process()
    train, test = train_test_split(data, test_size=0.2)
    no_of_movies = data[MOVIE_LABEL].nunique()
    no_of_users = data[USER_LABEL].nunique()
    data[USER_LABEL] = data[USER_LABEL] / data[USER_LABEL].max()
    model = get_user_model(no_of_movies, no_of_users)
    hist = model.fit(
        train[[MOVIE_LABEL, RATING]].values,
        train[[USER_LABEL]].values,
        batch_size=128,
        epochs=10,
        verbose=1,
        validation_data=(test[[MOVIE_LABEL, RATING]].values,
                         test[[USER_LABEL]].values)

    )
    model.save(f"{MODEL_NAME}_sub")
    save_history(hist, "fitting_sub.png")


def analyse():
    movies = [1, 15, 5, 10]
    ratings = [5, 2, 0, 5]
    model = keras.models.load_model(f"{MODEL_NAME}_sub")  # type: keras.Model
    data = np.asarray(movies)
    data = np.vstack((data, np.asarray(ratings))).T
    print(model.predict(data))


def run():
    # train_model()
    # analyse()
    train_similar_user_model()
