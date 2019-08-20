"""
Author : Rohit Suratekar
Data : July 2019

This file contains functions related to clustering

I am going to use DASK (https://dask.org/) here instead pandas
"""

import itertools

import dask.dataframe as dd

from helper.parser import *


def generate_tag_file():
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


def get_tag_association():
    with open(OUT_FOLDER + FILE_RAW_TAGS) as f:
        df = dd.read_csv(OUT_FOLDER + FILE_RAW_TAGS, sep="\t")
        # df = df.groupby([RAW_COL_1_TAG1, RAW_COL_2_TAG2]).count()
        print(df.head())


def test():
    d = pd.DataFrame({"a": ["a", "b", "c"], "b": ["b", "b", "c"]})
    print(d)


def run():
    test()
