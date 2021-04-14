import datetime
from sklearn.metrics import recall_score

from packages.gridsearch.helper import append_results_to_file, build_fragments, encode_fragments, \
    calc_number_combinations, parameter_generator
from packages.linear_model.MulticlassLogisticRegression import MulticlassLogisticRegression


def hyperparameter_generator(list_eta, list_epsilon, list_penalty, list_l2_lambda, list_max_iter):
    """

    :param list_eta:
    :param list_epsilon:
    :param list_penalty:
    :param list_l2_lambda:
    :param list_max_iter:
    :return:
    """
    for eta in list_eta:
        for e in list_epsilon:
            for penalty in list_penalty:
                for l2 in list_l2_lambda:
                    for m in list_max_iter:
                        yield eta, e, penalty, l2, m


def run_mlr_classification_recall(X_train, X_test, y_train, y_test, eta, epsilon, penalty, l2_lambda, max_iter):
    """
    Score is species level recall.

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param eta:
    :param epsilon:
    :param penalty:
    :param l2_lambda:
    :param max_iter:
    :return:
    """
    mlr = MulticlassLogisticRegression(eta=eta,
                                       epsilon=epsilon,
                                       penalty=penalty,
                                       l2_lambda=l2_lambda,
                                       max_iter=max_iter,
                                       verbose=True)
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
    """

    Todo - add ability to track runtime

    :param seq_file:
    :param taxid_file:
    :param output_dir:
    :param pattern:
    :param list_sample_length:
    :param list_coverage:
    :param list_k:
    :param list_eta:
    :param list_epsilon:
    :param list_penalty:
    :param list_l2_lambda:
    :param list_max_iter:
    :param seed:
    :param grid_search_file:
    :param fields:
    :param experiment:
    :param score_type:
    :return:
    """
    # set up grid search results file
    append_results_to_file(grid_search_file, fields=fields)

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
            append_results_to_file(grid_search_file, rows=[row])

        print('Percent complete: {}'.format(count / n_combinations * 100))  # display progress


def main():
    """
    Todo - change hardcoded parameters into parameters than can be supplied to method at command line
    :return:
    """
    # parameters
    seq_file = '/Users/ryanqnelson/GitHub/C-A-L-C-I-F-E-R/CMU-02620-Metagenomics/data/train_small-db_toy-3000.fasta'
    taxid_file = '/Users/ryanqnelson/GitHub/C-A-L-C-I-F-E-R/CMU-02620-Metagenomics/data/train_small-db_toy-3000.taxid'
    output_dir = '/Users/ryanqnelson/GitHub/C-A-L-C-I-F-E-R/CMU-02620-Metagenomics/data/sampling/sampling-toy-3000'
    pattern = 'fragments*.npy'
    seed = None
    date_time = datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')
    data_dir = '/Users/ryanqnelson/GitHub/C-A-L-C-I-F-E-R/CMU-02620-Metagenomics/'
    grid_search_file = data_dir + 'data/gridsearch-3000/results-3000-mlr.{}.csv'.format(date_time)
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

    experiment = '16.05'
    score_type = 'species_recall'

    # combinations to try
    list_sample_length = [100, 200, 400] * 5
    list_coverage = [1, 10, 100, 200, 400]
    list_k = [1, 2, 4, 6, 8, 10, 12]
    list_eta = [0.1]
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
