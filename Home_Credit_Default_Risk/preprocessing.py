"""
Define functions for data preprocessing
"""

import pandas as pd
import numpy as np


def print_long_string(string, indent=0, max_words_per_line=10):
    """
    :param string: str
    :param indent: int
    :param max_words_per_line: int
    :return: None
    """
    words = [" "*indent]
    for i, word in enumerate(string.split()):
        words.append(word)
        if (i+1) % max_words_per_line == 0:
            words.append("\n" + " "*indent)
    print(" ".join(words))
    return None

