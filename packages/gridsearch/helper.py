import csv
import shutil

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from packages.metagenomics import sampling2, encoding2


def append_results_to_file(filename, fields=None, rows=None):
    with open(filename, 'a') as f:

        write = csv.writer(f)

        if fields:
            write.writerow(fields)

        if rows:
            write.writerows(rows)


def build_fragments(seq_file, taxid_file, output_dir, sample_length, coverage, seed):
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
    Converts sparse matrix to array before splitting.
    """

    # encode data
    fragments = sampling2.read_fragments(output_dir, pattern)
    X_enc, y = encoding2.encode_fragment_dataset(fragments, k)
    le = preprocessing.LabelEncoder()
    y_enc = le.fit_transform(y)

    print('Encoded fragments...')
    print(X_enc.shape)

    # calculate number of classes
    n_classes = len(np.unique(y_enc))
    #     print('n_classes:',n_classes)
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
            msg = 'Not possible for both training and test sets to contain all classes.'
            msg2 = ' (n_classes, training set length, test set length):'
            raise ValueError(msg + msg2 + str(n_classes), len(y_train), len(y_test))

    print('Encoding succeeded.')
    return X_train, X_test, y_train, y_test


def calc_number_combinations(*args):
    total = 1
    for each in args:
        total *= len(each)
    return total


def parameter_generator(list_sample_length, list_coverage, list_k):
    for L in list_sample_length:
        for c in list_coverage:
            for k in list_k:
                yield L, c, k


def calc_hyperparameter_relationship(filename):
    """
    Runs logistic regression over hyperparameters to find the regression coefficients.
    This should give some indicator of how hyperparameters are affecting the score.
    """
    # read in grid search results
    df = pd.read_csv(filename)
    X = df.drop(['experiment', 'score', 'category', 'classifier'], axis=1)
    y = df['score']

    lr = LinearRegression()
    lr.fit(X, y)
    return lr.coef_