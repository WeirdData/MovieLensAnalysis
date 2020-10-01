#  Copyright (c) 2020.  WeirdData
#  Author: Rohit Suratekar
#
#  Dataset used: MovieLens 20M
#  https://www.kaggle.com/grouplens/movielens-20m-dataset
#
#  General statistics

import matplotlib.pyplot as plt
import pandas as pd
import pygraphviz as pgv
from SecretColors import Palette
from matplotlib import rc

from constants import *
from cosine_distance import get_movies

FILE_RATINGS = "data/rating.csv"  # Path to rating.csv
FILE_MOVIES = "data/movie.csv"  # Path to movie.csv
FILE_TAGS = "data/tag.csv"  # Path to tag.cdv


def get_stats():
    df = get_movies()
    df = df.fillna("")
    df[YEAR] = df[YEAR].map(
        lambda x: "".join([y for y in x if str(y).isnumeric()]))
    df = df[df[YEAR] != ""]
    print(f"Total Movies : {len(df)}")
    cats = df[MOVIE_GENRES].map(lambda x: x.split("|")).sum()
    print(f"Total Categories : {len(set(cats))}")
    print(f"From year {df[YEAR].min()} to {df[YEAR].max()}")
    df = pd.read_csv(FILE_RATINGS)
    print(f"Total ratings : {len(df)}")
    print(f"Total users : {df[USER_ID].nunique()}")


def plot_flow():
    p = Palette()
    g = pgv.AGraph(
        directed=True, dpi=150, pad=0.1,
        fontname="IBM Plex Sans")
    g.add_node("c", label="Tags, Categories and Ratings",
               shape="plaintext")
    g.add_node("b",
               label="<Distance Matrix<BR/> "
                     "<FONT POINT-SIZE='10'>(content based "
                     "filtering)</FONT>>",
               style="filled",
               margin="0.2",
               fillcolor=p.blue(shade=20),
               shape="box")
    g.add_node("a", label="Selected Movie",
               shape="plaintext")
    g.add_node("e", label="Similar Movies", shape="plaintext")
    g.add_node("d", label=r"Selected Movies\rand Ratings\r",
               shape="plaintext")
    g.add_node("f", label="<Neural Network<BR/>"
                          "<FONT POINT-SIZE='10'>(user based "
                          "filtering)</FONT>>",
               style="filled",
               margin="0.2",
               fillcolor=p.blue(shade=20),
               shape="box")
    g.add_node("g", label="Movies user\lmight like\l",
               shape="plaintext")
    g.add_node("h", label="Users and Ratings",
               shape="plaintext")

    g.add_edge("c", "b", style="dashed")
    g.add_edge("d", "f", minlen=3)
    g.add_edge("a", "b", minlen=3)
    g.add_edge("b", "e", minlen=3)
    g.add_edge("f", "g", minlen=3)
    g.add_edge("f", "h", dir="back", style="dashed")
    g.add_edge("b", "f", style="invis")

    g.add_subgraph(["a", "b", "e"], rank="same", name="s1",
                   label="ss")
    g.add_subgraph(["d", "f", "g"], rank="same", name="s2")

    g.layout("dot")
    g.draw("plot.png")


def plot_movies():
    p = Palette()
    plt.rcParams['axes.facecolor'] = p.gray(shade=10)
    df = get_movies()
    df = df.fillna("")
    df = df[df[YEAR] != ""]
    df[YEAR] = df[YEAR].map(
        lambda x: "".join([y for y in x if str(y).isnumeric()]))
    df[YEAR] = pd.to_numeric(df[YEAR])
    df = df[df[YEAR] < 2020]
    total_movies = len(df)
    df = df.groupby(by=YEAR).count().sort_index().reset_index()
    data = dict(zip(df[YEAR], df[MOVIE_ID]))
    plt.bar(range(0, len(data.keys())), data.values(), width=1,
            color=p.aqua(shade=40), zorder=3)
    labels = [int(x) for x in data.keys()][::5]
    plt.xticks(range(0, len(labels) * 5, 5), labels, rotation=90)
    plt.ylabel("Frequency")
    plt.xlabel("Year", labelpad=10)
    plt.grid(ls="--", color=p.gray(shade=30), zorder=-1)
    rc('text', usetex=True)
    plt.annotate(f"MovieLens Dataset\n"
                 r"\small{(Total movies: " + str(total_movies) +
                 r")}",
                 (0.1, 0.8), xycoords="axes fraction",
                 fontsize=13,
                 bbox=dict(fc=p.white(), lw=0.1, pad=10))

    plt.tight_layout()
    plt.savefig("plot.png", dpi=150)
    plt.show()


def run():
    plot_flow()
