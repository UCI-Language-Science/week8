# Starter code for HW8

# N.B.: you may need to run the following:
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('treebank')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.tree import Tree
from nltk.corpus import treebank

from typing import Iterable

##############################################################################
# Part I
##############################################################################

def tagged_sentences(corpus: list[str]) -> list[list[tuple[str]]]:
    """Given a list of sentences, return the list of the sentences tokenized, using nltk.pos_tag and nltk.tokenize.word_tokenize."""
    # TODO: your implementation here, about 1 line


def get_pos_tag_observations(corpus: list[str]) -> pd.DataFrame:
    """Given a list of sentences, return a pandas DataFrame with one word per row, containing columns for its pos, the index of the word in the sentence, and the index of the sentence in the corpus. 
    
    If you print the dataframe, the beginning should look something like this:

                    word  pos  word_num  sent_num
        0        Pierre  NNP         0         0
        1        Vinken  NNP         1         0
        2             ,    ,         2         0
    """
    # TODO: your implementation here, about 1-3 lines


def pos_tags_sorted(corpus: list[str]) -> pd.Series:
    """Helper function you may find useful for `get_n_most_common_pos_tags` and `rarest_pos_tag_word`."""
    # TODO: Your implementation here, about 1 line if you have the dataframe of pos tags

def get_n_most_common_pos_tags(corpus: list[str], n: int) -> list[str]:
    """Given a list of sentences, return the n most common pos tags."""
    # TODO: your implementation here, about 1 line if you've implemented pos_tags_sorted


def rarest_pos_tag_word(corpus: list[str]) -> str:
    """Given a list of sentences, return the corresponding word in the corpus."""
    # Hint: use `get_pos_tag_observations` and sort
    # TODO: Your implementation here, about 3 lines

def plot_10_most_common_pos_tags(corpus: list[str]) -> None:
    """Given an iterable of parsed sentences, plot a histogram of the 10 most common pos tags. Save your plot to a file called 'histogram.png'."""
    # TODO: Your implementation here, 1 line
    plt.xlabel("Part of Speech Category")
    plt.ylabel("Count")
    plt.savefig('histogram.png', dpi=300, bbox_inches='tight')
    plt.close("all")

##############################################################################
# Part II
##############################################################################

from scipy.stats import linregress

def get_heights(parsed_sentences: Iterable) -> np.ndarray:
    """Given an iterable of parsed sentences, return an array containing the height (depth) of their syntax trees."""
    # TODO: your implementation here, about 1 line

def get_leaves(parsed_sentences: Iterable) -> np.ndarray:
    """Given an iterable of parsed sentences, return an array containing the number of leaves of their syntax trees."""
    # TODO: your implementation here, about 1 line

def predict_height(parsed_sentences: Iterable, num_leaves: int) -> float:
    """Given an iterable of parsed sentences, predict the height of a tree with `num_leaves` leaves."""
    # TODO: your implementation here, about 1-2 lines

def plot_regression(parsed_sentences: Iterable) -> None:
    """Given an iterable of parsed sentences, plot a linear regression of tree leaves (x-axis) and tree heights (y-axis). Save your plot to a file to disk called 'regression.png'."""
    # TODO: your implementation here, about 5-7 lines
    # Hint: most of this is just repeating code from `predict_height`

    plt.xlabel("Number of leaves")
    plt.ylabel("Tree depth")
    plt.savefig('regression.png', dpi=300, bbox_inches='tight')
    plt.close("all")
