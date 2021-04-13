"""
Common functions used in grid search process.
"""

import csv
import shutil

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from packages.metagenomics import sampling2, encoding2


def append_results_to_file(filename, fields=None, rows=None):
    """
    Appends fields and/or rows to given file in csv format.

    :param filename: path to the file
    :param fields: List, header row for the file
    :param rows: List, row(s) of data to be written to file
    :return: None
    """
    with open(filename, 'a') as f:

        write = csv.writer(f)

        if fields:
            write.writerow(fields)

        if rows:
            write.writerows(rows)


def build_fragments(seq_file, taxid_file, output_dir, sample_length, coverage, seed):
    """
    Deletes output directory if it exists. Populates output directory with fragment data.

    :param seq_file: Path to file containing sequence data in .fasta format.
    :param taxid_file: Path to file containing matching species for all sequences.
    :param output_dir: Path where fragments should be written. Directory will be deleted if it exists.
    :param sample_length: int, Length of the fragments to extract from each sequence.
    :param coverage: float, desired coverage percent for each sequence letter
    :param seed: Random seed, for reproducibility
    :return: None
    """
    # delete output directory if it previously exists
    try:
        shutil.rmtree(output_dir)
    except FileNotFoundError:
        print('Existing directory was not found. Process will generate a directory.')

    # build fragments
    print('Building fragments...')
    sampling2.generate_fragment_data(seq_file, taxid_file, output_dir, sample_length, coverage, seed)


def encode_fragments(output_dir, pattern, k, seed=None):
    """
    Reads fragment data from file, encodes data for processing, and splits data into training and test sets.
    Performs an additional check to ensure that both test and training sets contain all classes in the data.

    :param output_dir: Path where fragments were written.
    :param pattern: str, bash-like pattern defining types of files to read from the output directory.
                (i.e. "*.npy" to read all files that end with .npy)
    :param k: int, size of k-mer to subdivide fragments into
    :param seed: Random seed, for reproducibility
    :return: L x J sparse matrix, where L is the number of fragments and J is the number of dimensions
            for each fragment.
    """

    # encode data and labels
    fragments = sampling2.read_fragments(output_dir, pattern)
    X_enc, y = encoding2.encode_fragment_dataset(fragments, k)
    le = preprocessing.LabelEncoder()
    y_enc = le.fit_transform(y)

    print('Encoded fragments...')
    print(X_enc.shape)

    # perform check so that randomly split training and test sets both contain all classes in the data
    n_classes = len(np.unique(y_enc))
    n_classes_train = 0
    n_classes_test = 0
    X_train, X_test, y_train, y_test = None, None, None, None
    count = 0
    while n_classes_train < n_classes or n_classes_test < n_classes:
        if n_classes_train != 0:
            print('Encoding failed')

        # split data into test and training
        X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.33, random_state=seed)
        n_classes_train = len(np.unique(y_train))
        n_classes_test = len(np.unique(y_test))
        count += 1

        if count > 1000:
            # there must be an issue and we are stuck in an infinite loop
            msg = 'Not possible for both training and test sets to contain all classes.'
            msg2 = ' (n_classes, training set length, test set length):'
            raise ValueError(msg + msg2 + str(n_classes), len(y_train), len(y_test))

    print('Encoding succeeded.')
    return X_train, X_test, y_train, y_test


def calc_number_combinations(*args):
    """
    Determines the number of parameter combinations.

    :param args: any number of collections
    :return:
    """
    total = 1
    for each in args:
        total *= len(each)  # Multiplies lengths of all collections together.
    return total


def parameter_generator(list_sample_length, list_coverage, list_k):
    """
    Builds generator for main parameter combinations.

    :param list_sample_length: List, sample lengths to be tested
    :param list_coverage: List, coverages to be tested
    :param list_k: List, k-mers to be tested
    :return: Single (sample length, coverage, k) combination each time generator is called.
    """
    for L in list_sample_length:
        for c in list_coverage:
            for k in list_k:
                yield L, c, k


def calc_hyperparameter_relationship(filename, droplist, score_col):
    """
    Runs linear regression over hyperparameters to find the regression coefficients.
    This should give some indicator of how hyperparameters are affecting the score.

    :param filename: Path to file containing search results.
    :param droplist: All columns to drop from the dataset before performing linear regression.
                    (i.e. ['experiment', 'score', 'category', 'classifier'])
    :return: coefficients for each of the remaining feature columns
    """
    # read in grid search results
    df = pd.read_csv(filename)
    X = df.drop(droplist, axis=1)
    y = df[score_col]

    lr = LinearRegression()
    lr.fit(X, y)
    return lr.coef_
