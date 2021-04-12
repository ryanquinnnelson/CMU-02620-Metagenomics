import shutil
import csv
import datetime
import numpy as np
from sklearn.metrics import recall_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from packages.metagenomics import sampling2, encoding2
from packages.LogisticRegression.MulticlassLogisticRegression import MulticlassLogisticRegression2


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


def encode_fragments(output_dir, pattern, k, seed):
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
    while n_classes_train < n_classes or n_classes_test < n_classes:
        print('Encoding failed')

        # split data into test and training
        X_train, X_test, y_train, y_test = train_test_split(X_enc.toarray(), y_enc, test_size=0.33)
        n_classes_train = len(np.unique(y_train))
        n_classes_test = len(np.unique(y_test))
    #         print('train:',len(y_train))
    #         print('test:', len(y_test))

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


def hyperparameter_generator(list_eta, list_epsilon, list_penalty, list_l2_lambda, list_max_iter):
    for eta in list_eta:
        for e in list_epsilon:
            for penalty in list_penalty:
                for l2 in list_l2_lambda:
                    for m in list_max_iter:
                        yield eta, e, penalty, l2, m


def run_mlr_classification_recall(X_train, X_test, y_train, y_test, eta, epsilon, penalty, l2_lambda, max_iter):
    """
    Score is species level recall.
    """
    mlr = MulticlassLogisticRegression2(eta=eta,
                                        epsilon=epsilon,
                                        penalty=penalty,
                                        l2_lambda=l2_lambda,
                                        max_iter=max_iter)
    mlr.fit(X_train, y_train)
    y_pred = mlr.predict(X_test)
    score = recall_score(y_test, y_pred, average='weighted')
    return score


def grid_search_multiclass_mlr(seq_file,
                               taxid_file,
                               output_dir,
                               pattern,
                               list_sample_length,
                               list_coverage,
                               list_k,
                               list_eta,
                               list_epsilon,
                               list_penalty,
                               list_l2_lambda,
                               list_max_iter,
                               seed,
                               grid_search_file,
                               fields,
                               experiment,
                               score_type):
    # set up grid search results file
    append_results_to_file(grid_search_file, fields)

    # calculate number of combinations
    n_combinations = calc_number_combinations(list_sample_length,
                                              list_coverage,
                                              list_k,
                                              list_eta,
                                              list_epsilon,
                                              list_penalty,
                                              list_l2_lambda,
                                              list_max_iter)

    # process combinations
    count = 0
    sample_length_prev = -1
    coverage_prev = -1

    # parameter combinations
    for sample_length, coverage, k in parameter_generator(list_sample_length, list_coverage, list_k):
        print(sample_length, coverage, k)

        if sample_length != sample_length_prev or coverage != coverage_prev:
            # fragment combination
            build_fragments(seq_file, taxid_file, output_dir, sample_length, coverage, seed)

            # update previous values
            sample_length_prev = sample_length
            coverage_prev = coverage

        # kmer from fragments
        X_train, X_test, y_train, y_test = encode_fragments(output_dir, pattern, k, seed)

        # hyperparameter combinations
        for eta, epsilon, penalty, l2_lambda, max_iter in hyperparameter_generator(list_eta, list_epsilon, list_penalty,
                                                                                   list_l2_lambda, list_max_iter):
            print(eta, epsilon, penalty, l2_lambda, max_iter)

            # train and score model
            score = run_mlr_classification_recall(X_train, X_test, y_train, y_test, eta, epsilon, penalty, l2_lambda,
                                                  max_iter)
            count += 1

            # output results to file
            row = [experiment, 'multiclass', 'Logistic Regression', X_train.shape, sample_length, coverage, k, eta,
                   epsilon, penalty, l2_lambda, max_iter, score, score_type]
            append_results_to_file(grid_search_file, row)

        print('Percent complete: {}'.format(count / n_combinations * 100))  # display progress


def main():
    # parameters
    seq_file = '/Users/ryanqnelson/GitHub/C-A-L-C-I-F-E-R/CMU-02620-Metagenomics/data/train_small-db_toy-3000.fasta'
    taxid_file = '/Users/ryanqnelson/GitHub/C-A-L-C-I-F-E-R/CMU-02620-Metagenomics/data/train_small-db_toy-3000.taxid'
    output_dir = '/Users/ryanqnelson/GitHub/C-A-L-C-I-F-E-R/CMU-02620-Metagenomics/data/sampling/sampling-toy-3000'
    pattern = 'fragments*.npy'
    seed = 42
    date_time = datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')
    grid_search_file = '/Users/ryanqnelson/GitHub/C-A-L-C-I-F-E-R/CMU-02620-Metagenomics/data/gridsearch-3000/results-3000-mlr.{}.csv'.format(
        date_time)
    fields = ['experiment',
              'category',
              'classifier',
              'training shape',
              'sample_length',
              'coverage',
              'k',
              'eta',
              'epsilon',
              'penalty',
              'l2_lambda',
              'max_iter',
              'score',
              'score type']

    experiment = '15.01'
    score_type = 'species_recall'

    # combinations to try
    list_sample_length = [200]  # ,200,400
    list_coverage = [100]  # 1,10,100,
    list_k = [6]  # ,8,10,12
    list_eta = [0.05]
    list_epsilon = [0.01]
    list_penalty = [None]
    list_l2_lambda = [0]
    list_max_iter = [200]

    grid_search_multiclass_mlr(seq_file,
                               taxid_file,
                               output_dir,
                               pattern,
                               list_sample_length,
                               list_coverage,
                               list_k,
                               list_eta,
                               list_epsilon,
                               list_penalty,
                               list_l2_lambda,
                               list_max_iter,
                               seed,
                               grid_search_file,
                               fields,
                               experiment,
                               score_type)


if __name__ == "__main__":
    main()
