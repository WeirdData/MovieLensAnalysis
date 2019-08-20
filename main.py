"""
Author : Rohit Suratekar
Data : July 2019

Data analysis on the MovieLens data-set

( https://grouplens.org/datasets/movielens/ )

Acknowledgment :

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Transactions  on Interactive Intelligent Systems (TiiS) 5,
4, Article 19 (December 2015),  19 pages.
DOI=<http://dx.doi.org/10.1145/2827872>

Dataset was accessed on 27 July 2019

"""

import os

from analysis.clustering import run
from constants.system import SAVE_FOLDER, OUT_FOLDER

# Create folder for saving plots
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Create folder for saving output files
if not os.path.exists(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)
run()
