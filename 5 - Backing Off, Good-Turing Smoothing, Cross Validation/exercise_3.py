import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from typing import List


def preprocess(text: List[str]) -> List[str]:
    """ Preprocess the input text by removing extra spaces, characters and lowercasing.
    Args:
        text: The input text to preprocess.

    Returns:
        list: A list of tokens (words) after preprocessing.
    """
    words = []
    for sentence in text:
        sentence = sentence.strip()
        sentence = sentence.lower()

        curr_word=''
        for char in sentence:
            if char.isalpha(): #checks if alphabetical
                curr_word+=char
            else:
                if curr_word:
                    words.append(curr_word)
                    curr_word = ''

        if curr_word:
            words.append(curr_word)

    return words


def train_test_split_data(text: List[str], test_size=0.2):
    """ Splits the input corpus in a train and a test set
    Args:
        text: input corpus
        test_size: size of the test set, in fractions of the original corpus

    Returns: 
        train and test set
    """
    train,test= train_test_split(text, test_size=test_size, random_state=42)
    return train,test


def k_validation_folds(text: List[str], k_folds=10):
    """ Splits a corpus into k_folds cross-validation folds
        text: input corpus
        k_folds: number of cross-validation folds

    Returns: 
        the cross-validation folds
    """
    fold_size = len(text) // k_folds
    folds = []
    
    for i in range(k_folds):
        start = i * fold_size
        # For the last fold, include all remaining tokens to avoid losing any tokens
        end = (i + 1) * fold_size if i < k_folds - 1 else len(text)
        folds.append(text[start:end])
    
    return folds


def plot_pp_vs_alpha(pps: List[float], alphas: List[float], N: int):
    """ Plots n-gram perplexity vs alpha
    Args:
        pps: list of perplexity scores
        alphas: list of alphas
        N: just for plotting
    """
    plt.figure(figsize=(8,5))
    plt.plot(alphas, pps, marker='o')
    plt.xlabel("Alpha (Smoothing parameter)")
    plt.ylabel("Perplexity")
    plt.title(f"{N}-gram Language Model Perplexity vs Alpha")
    plt.grid(True)
    plt.show()
