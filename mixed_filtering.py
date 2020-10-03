#  Copyright (c) 2020.  WeirdData
#  Author: Rohit Suratekar
#
#  Dataset used: MovieLens 20M
#  https://www.kaggle.com/grouplens/movielens-20m-dataset
#
#  Movie prediction based on neural network

from common import get_movies
from constants import *
from sklearn.feature_extraction.text import CountVectorizer


def prepare_data():
    movies = get_movies()
    print(movies)


def run():
    prepare_data()
