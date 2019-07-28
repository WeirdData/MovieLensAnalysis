"""
Data Parsers will go in this file
"""

import pandas as pd

from constants.dataset import *
from constants.system import *


def get_movies() -> pd.DataFrame:
    """
    :return: Pandas Dataset of Movies with additional column
    """
    # Get Data
    with open(DATA_FOLDER + FILE_MOVIES) as f:
        d = pd.read_csv(f)
    # Isolate the year
    d[MOVIE_COL_4_YEAR] = d[MOVIE_COL_2_TITLE] \
        .apply(lambda x: x[-6:].strip().replace("(", "").replace(")", ""))

    return d


def get_ratings() -> pd.DataFrame:
    """
    Ratings are on the scale of 0.5 to 5 with 0.5 as an increment
    :return: Pandas DataFrame of rating data
    """
    with open(DATA_FOLDER + FILE_RATINGS) as f:
        return pd.read_csv(f)


def get_tags() -> pd.DataFrame:
    """
    Each line of this file after the header row represents one tag applied to
    one movie by one user.

    :return: Pandas DataFrame of tags
    """
    with open(DATA_FOLDER + FILE_TAGS) as f:
        return pd.read_csv(f)


def run():
    d = get_tags()
    print(d)
