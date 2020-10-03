#  Copyright (c) 2020.  WeirdData
#  Author: Rohit Suratekar
#
#  Dataset used: MovieLens 20M
#  https://www.kaggle.com/grouplens/movielens-20m-dataset
#
# Common functions which can be used anywhere in the repository

import pandas as pd
from constants import *


def get_movies() -> pd.DataFrame:
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
            # Correct for 'the'
            if ", The" in m:
                return f"The {_temp[0].strip().replace(', The', '')}"
            return _temp[0].strip()

    df = pd.read_csv("data/movie.csv")
    # Remove where Genres are not listed
    df = df[df[MOVIE_GENRES] != "(no genres listed)"]
    df[YEAR] = df[MOVIE_TITLE].map(lambda x: _separate_year(x))
    df[MOVIE_TITLE] = df[MOVIE_TITLE].map(lambda x: _separate_title(x))
    return df.reset_index(drop=True)
