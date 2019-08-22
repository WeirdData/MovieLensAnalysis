"""
Author : Rohit Suratekar
Data : July 2019

This file contains functions related to clustering

Be careful about running functions from this script. These need more memory
than normal functions. I have tested them successfully on 12 GB RAM with i7
processor.
"""

import itertools

from helper.parser import *


def generate_full_tag_file():
    """
    Generates raw file containing tags association.

    In this function we have to open file without standard 'with' argument
    to avoid any memory problem. It will generate file with approximately
    2.4 GB
    """
    data = get_tags()
    data = data[[TAG_COL_2_MOVIE_ID, TAG_COL_3_TAG]]
    # Isolate the tags which are not unique
    data = data[data.duplicated(subset=[TAG_COL_3_TAG], keep=False)]
    data = data.groupby(TAG_COL_2_MOVIE_ID)

    file = open(OUT_FOLDER + FILE_RAW_TAGS, "w")
    file.write("{}\t{}\n".format(RAW_COL_1_TAG1, RAW_COL_2_TAG2))
    for d in data:
        all_tags = d[1][TAG_COL_3_TAG].values
        all_tags = itertools.product(all_tags, all_tags)
        for i in all_tags:
            if i[0] != i[1]:
                file.write("{}\t{}\n".format(i[0], i[1]))

        file.flush()

    # Dont forget to close the file
    file.close()


def generate_tag_association_file():
    with open(OUT_FOLDER + FILE_RAW_TAGS) as f:
        df = pd.read_csv(OUT_FOLDER + FILE_RAW_TAGS, delimiter="\t")
        df = df.groupby(df.columns.tolist(),
                        as_index=False).size().reset_index()
        df = df.rename(columns={0: RAW_COL_OCCURRENCE})
    tmp = "tmp"
    df[tmp] = df.apply(lambda x: " ".join(sorted([x[RAW_COL_2_TAG2],
                                                  x[RAW_COL_1_TAG1]])),
                       axis=1)
    df = df.drop_duplicates(subset=[tmp])
    del df[tmp]

    with open(OUT_FOLDER + FILE_JOINED, "w") as f:
        print(df.to_csv(sep="\t", index=False), file=f)


def get_association():
    with open(OUT_FOLDER + FILE_JOINED) as f:
        data = pd.read_csv(f, delimiter="\t")

    data = data.sort_values(by=RAW_COL_OCCURRENCE,
                            ascending=False).reset_index(drop=True)
    return data


def search_association(terms: list):
    data = get_association()
    data = data[data[RAW_COL_1_TAG1].isin(terms)]
    data = data[data[RAW_COL_2_TAG2].isin(terms)]
    print(data)


def run():
    search_association(["action", "sci-fi"])
