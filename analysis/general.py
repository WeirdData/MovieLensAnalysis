"""
Author : Rohit Suratekar
Data : July 2019

This file contains general functions used in calculating and visualizing
MovieLens dataset
"""
import matplotlib.pylab as plt
import numpy as np
from SecretColors.palette import Palette

from helper.parser import *


def basic_statistics():
    movies = get_movies()
    ratings = get_ratings()
    tags = get_tags()

    print("Number of Movies {}".format(len(movies)))
    print("Number of Total Ratings {}".format(len(ratings)))
    print("Number of Total Tag {}".format(len(tags)))

    # All users are arranged in descending order of their IDs, so just check
    # ID of the last user to know number of users
    print("Number of Users Rated and Tagged {}"
          .format(ratings[RATING_COL_1_USER_ID].iloc[-1]))


def movie_ratings():
    movies = get_movies()
    ratings = get_ratings()
    p = Palette()
    # Extract rating

    print("Top movies rated by users ")

    # In following example, 'userID' column actually shows number of users
    # who rated that movie
    r = ratings.groupby([RATING_COL_2_MOVIE]).agg({
        RATING_COL_1_USER_ID: "count", RATING_COL_3_RATING: "sum"})

    r[RATING_COL_5_EXTRA] = r[RATING_COL_3_RATING] / r[RATING_COL_1_USER_ID]
    r = r.sort_values(by=RATING_COL_1_USER_ID, ascending=False)
    r = pd.merge(r, movies, on=RATING_COL_2_MOVIE)
    r = r[[MOVIE_COL_2_TITLE, RATING_COL_1_USER_ID, RATING_COL_2_MOVIE,
           RATING_COL_3_RATING, RATING_COL_5_EXTRA]].iloc[:5]
    print(r)

    x_ticks = np.arange(0, 6, 0.5)
    plt.hist(ratings[RATING_COL_3_RATING], bins=x_ticks - 0.25,
             color=p.orange(shade=40), rwidth=0.8, zorder=2)
    plt.xlabel("User Rating")
    plt.ylabel("Frequency")
    plt.title("Individual User Ratings")
    plt.axvline(round(ratings[RATING_COL_3_RATING].mean(), 1), linestyle="--",
                color=p.black(), label="Average Rating Given", zorder=3)
    plt.xticks(x_ticks[:-1])
    plt.legend(loc=0)
    plt.xlim(0, 5.5)
    plt.grid(zorder=0, color=p.gray(shade=30))
    plt.gca().patch.set_facecolor(p.gray(shade=10))
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(-2, 2),
                               useMathText=True)
    plt.tight_layout()
    plt.savefig("{}/UserRatings.png".format(SAVE_FOLDER), format='png',
                dpi=300)
    plt.show()


def run():
    movie_ratings()
