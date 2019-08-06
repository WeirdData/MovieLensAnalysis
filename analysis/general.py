"""
Author : Rohit Suratekar
Data : July 2019

This file contains general functions used in calculating and visualizing
MovieLens dataset
"""
from collections import Counter

import matplotlib.pylab as plt
import numpy as np
from SecretColors import Palette

from helper.parser import *


def basic_statistics():
    """
    Basic statistics regarding movies, rating and tags
    """
    movies = get_movies()
    ratings = get_ratings()
    tags = get_tags()

    print("Number of Movies {}".format(len(movies)))
    print("Earliest movie from {} while latest movie from {}"
          .format(movies[MOVIE_COL_4_YEAR].min(),
                  movies[MOVIE_COL_4_YEAR].max()))
    print("Number of Total Ratings {}".format(len(ratings)))
    print("Number of Total Tag {}".format(len(tags)))

    # All users are arranged in descending order of their IDs, so just check
    # ID of the last user to know number of users
    print("Number of Users Rated and Tagged {}"
          .format(ratings[RATING_COL_1_USER_ID].iloc[-1]))


def movie_ratings():
    """
    Shows top movies rated by users and plots histogram of average rating
    given by users
    """
    movies = get_movies()
    ratings = get_ratings()
    p = Palette()
    # Extract rating

    print("Top movies rated by users. 'userID' column here shows number of "
          "users who voted that movie while 'extra' shows the average rating "
          "of the movie\n")

    # In following example, 'userID' column actually shows number of users
    # who rated that movie
    r = ratings.groupby([RATING_COL_2_MOVIE]).agg({
        RATING_COL_1_USER_ID: "count", RATING_COL_3_RATING: "sum"})

    r[RATING_COL_5_EXTRA] = r[RATING_COL_3_RATING] / r[RATING_COL_1_USER_ID]
    r = r.sort_values(by=RATING_COL_1_USER_ID, ascending=False)
    r = pd.merge(r, movies, on=RATING_COL_2_MOVIE)
    r = r[[MOVIE_COL_2_TITLE, RATING_COL_1_USER_ID, RATING_COL_2_MOVIE,
           RATING_COL_3_RATING, RATING_COL_5_EXTRA]]
    print(r.iloc[:5])

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


def movie_rating_trend():
    """
    Checks trend between movie ratings and users
    """
    movies = get_movies()
    ratings = get_ratings()
    p = Palette()
    # Extract rating

    # In following example, 'userID' column actually shows number of users
    # who rated that movie
    r = ratings.groupby([RATING_COL_2_MOVIE]).agg({
        RATING_COL_1_USER_ID: "count", RATING_COL_3_RATING: "sum"})

    r[RATING_COL_5_EXTRA] = r[RATING_COL_3_RATING] / r[RATING_COL_1_USER_ID]
    r = r.sort_values(by=RATING_COL_1_USER_ID, ascending=False)
    r = pd.merge(r, movies, on=RATING_COL_2_MOVIE)
    r = r[[MOVIE_COL_2_TITLE, RATING_COL_1_USER_ID, RATING_COL_2_MOVIE,
           RATING_COL_3_RATING, RATING_COL_5_EXTRA]]

    user_cutoff = 13568
    k = r[r[RATING_COL_1_USER_ID] > 0][RATING_COL_1_USER_ID]
    # Due to long tail of this histogram, plot only which can below cutoff
    plt.hist(k, 100, color=p.aqua())
    plt.ylabel("Frequency")
    plt.xlabel("Number of user ratings received by movie (Log Scale)")
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(-2, 2),
                               useMathText=True)

    # plt.yscale("log")

    qnt_loc = [0.25, 0.5, 0.75]
    markers = [':', '-.', "--"]
    qnt = np.quantile(r[RATING_COL_1_USER_ID], qnt_loc)

    for i, q in enumerate(qnt):
        plt.axvline(q, label="{} quantile".format(qnt_loc[i]),
                    color=p.black(), linestyle=markers[i])

    plt.legend(loc=0)
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig("{}/RatingsStats.png".format(SAVE_FOLDER), format='png',
                dpi=300)
    plt.show()


def movie_ratings_year_wise():
    """
    Plots year wise distribution of various aspects of movies and their ratings
    """
    p = Palette()
    movies = get_movies()
    ratings = get_ratings()

    r = ratings.groupby([MOVIE_COL_1_ID]).count()
    r[RATING_COL_6_RATING_COUNT] = r[RATING_COL_1_USER_ID]
    r = r[[RATING_COL_6_RATING_COUNT]]

    k = pd.merge(r, movies, on=MOVIE_COL_1_ID).set_index(MOVIE_COL_1_ID)
    k = k[[RATING_COL_6_RATING_COUNT, MOVIE_COL_4_YEAR]]
    k = k.reset_index()
    k = k.groupby([MOVIE_COL_4_YEAR]).agg({MOVIE_COL_1_ID: "count",
                                           RATING_COL_6_RATING_COUNT: "sum"})

    k[RATING_COL_5_EXTRA] = k[RATING_COL_6_RATING_COUNT] / k[MOVIE_COL_1_ID]

    k = k.sort_values(by=MOVIE_COL_4_YEAR)
    k[RATING_COL_6_RATING_COUNT] = k[RATING_COL_6_RATING_COUNT] / 1000
    k = k.reset_index()

    ind = range(len(k))

    plt.bar(ind, k[MOVIE_COL_1_ID], color=p.aqua(), width=1.0,
            label="No. of Movies")
    plt.bar(ind, k[RATING_COL_5_EXTRA], color=p.red(shade=70), alpha=0.7,
            width=1.0, label="Ratings per movie")

    plt.bar(ind, k[RATING_COL_6_RATING_COUNT], color=p.blue(),
            alpha=0.7,
            width=1.0, label="No. of ratings x $10^{3}$")

    plt.xticks(ind[1::2], k[MOVIE_COL_4_YEAR][1::2], rotation="90")
    # plt.yscale("log")
    plt.legend(loc=0)
    plt.xlabel("Year")
    plt.ylabel("Frequency")
    # plt.grid(axis="x", color=p.red(shade=20))
    plt.tight_layout()
    plt.savefig("{}/MovieAndRatingYear.png".format(SAVE_FOLDER), format='png',
                dpi=300)
    plt.show()


def top_tags():
    """
    Prints top tags used by the users and plots top categories assigned for
    the movies
    """
    p = Palette()
    tags = get_tags()
    tags = tags.groupby(by=TAG_COL_3_TAG).count().sort_values(
        by=TAG_COL_1_USER_ID, ascending=False)
    print("Top tags given by users")
    print(tags[TAG_COL_1_USER_ID].head())

    movies = get_movies()
    movies[MOVIE_COL_3_GENRES] = movies[MOVIE_COL_3_GENRES].str.split("|")
    movies = movies[MOVIE_COL_3_GENRES]
    c = Counter(movies.sum())
    names = c.keys()
    values = c.values()

    zipped = sorted(zip(values, names))
    values, names = zip(*zipped)
    ind = range(len(values))
    plt.barh(ind, values, color=p.aqua())
    plt.yticks(ind, names)
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig("{}/TopTags.png".format(SAVE_FOLDER), format='png',
                dpi=300)
    plt.show()


def movie_category_tags(category: str):
    """
    Check top tags for given category
    :param category: Name of the category as present in the file
    """
    movies = get_movies()
    tag = get_tags()
    movies = movies[movies[MOVIE_COL_3_GENRES].str.contains(category)]
    movies = movies.reset_index(drop=True)
    z = pd.merge(movies, tag, on=MOVIE_COL_1_ID)
    print("Top tags for the category: {}".format(category))
    print(z[TAG_COL_3_TAG].value_counts().head(10))


def run():
    # movie_and_ratings()
    # d = get_movies()
    movie_category_tags("Sci-Fi")
    # top_tags()
