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
from common import get_movies

FILE_RATINGS = "data/rating.csv"  # Path to rating.csv
FILE_MOVIES = "data/movie.csv"  # Path to movie.csv
FILE_TAGS = "data/tag.csv"  # Path to tag.cdv


def get_stats():
    """
     Generates basic statistics of the MovieLens database
     """
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
    """
    Generates the flow diagram of the Content-based and user-based filtering
    """
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
    """
    Generates histogram of movies available in MovieLens Database
    """
    p = Palette()
    plt.rcParams['axes.facecolor'] = p.gray(shade=10)
    df = get_movies()
    df = df.fillna("")
    df = df[df[YEAR] != ""]
    df[YEAR] = df[YEAR].map(
        lambda x: "".join([y for y in x if str(y).isnumeric()]))
    df[YEAR] = pd.to_numeric(df[YEAR])
    df = df[df[YEAR] < 3020]
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


def _generate_label(main_label, sub_label, words=4):
    n = f"<{main_label}<BR/><FONT POINT-SIZE='10'>"
    c = 0
    for m in sub_label.split(" "):
        n += f" {m}"
        if c == words:
            n += " <BR/> "
            c = 0
        else:
            c += 1

    return f"{n}</FONT>>"


def draw_distance_flow():
    """
    Draws flow diagram of the content-based filtering
    """
    p = Palette()
    g = pgv.AGraph(
        dpi=150, pad=0.1,
        directed=True, fontname="IBM Plex Sans"
    )

    g.node_attr["style"] = "filled"
    g.node_attr["shape"] = "box"
    g.node_attr["fillcolor"] = p.green(shade=20)
    g.node_attr["margin"] = "0.1"

    g.add_node("a",
               label=_generate_label(
                   "LabelEncoder",
                   "converts arbitrary words to categorical variables", 3))

    g.add_node("b", label=_generate_label(
        "CountVector",
        "counts the number of words of each category"))

    g.add_node("c1", label="tags", shape="plain", fillcolor=None)
    g.add_node("c2", label="categories", shape="plain", fillcolor=None)
    g.add_node("c3", label="rating", shape="plain", fillcolor=None)

    g.add_node("d", label=_generate_label(
        "Distance Matrix", "cosine distance"))

    g.add_node("e", label=_generate_label(
        "Sort and Filter", "according to distance w.r.t. input", 2))

    g.add_node("f", label="Prediction", fillcolor=p.blue(shade=30))

    g.add_edge("c", "a", minlen=3)
    g.add_edge("a", "b")
    g.add_edge("b", "d")
    g.add_edge("d", "e")
    g.add_edge("e", "f", minlen=3)

    g.add_edge("c1", "c")
    g.add_edge("c2", "c")
    g.add_edge("c3", "c")
    g.add_node("c", label="Bag of Words", fillcolor=p.blue(shade=30))
    g.add_node("c")

    g.add_subgraph(["c", "a"], rank="same")
    g.add_subgraph(["e", "f"], rank="same")
    g.layout("dot")
    g.draw("plot.png")


def draw_network_flow():
    """
    Draws flow diagram for user-based filtering
    """
    p = Palette()
    g = pgv.AGraph(
        dpi=150, pad=0.1, rankdir="LR",
        directed=True, fontname="IBM Plex Sans"
    )

    g.node_attr["style"] = "filled"
    g.node_attr["shape"] = "box"
    g.node_attr["fillcolor"] = p.green(shade=20)
    g.node_attr["margin"] = "0.1"
    g.node_attr["height"] = 0.9

    g.add_node("a1", label="user-id", shape="plain", fillcolor=None)
    g.add_node("a2", label="movie-id", shape="plain", fillcolor=None)

    g.add_node("a",
               label=_generate_label(
                   "User-Movie Matrix",
                   "Generates vectors representing user and movies", 3),
               fillcolor=p.blue(shade=30))
    g.add_node("b",
               label="Neural Network")

    g.add_node("c",
               label=_generate_label(
                   "Prediction",
                   "Rating of input movie based on user's rating history", 3),
               fillcolor=p.blue(shade=30))

    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("a1", "a")
    g.add_edge("a2", "a")
    g.layout("dot")
    g.draw("plot.png")


def run():
    draw_network_flow()
