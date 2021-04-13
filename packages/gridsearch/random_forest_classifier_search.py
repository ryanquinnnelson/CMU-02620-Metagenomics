import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score

from packages.gridsearch.helper import append_results_to_file, build_fragments, encode_fragments, \
    calc_number_combinations, parameter_generator


def hyperparameter_generator(list_max_depth, list_n_estimators):
    """

    :param list_max_depth:
    :param list_n_estimators:
    :return:
    """
    for max_depth in list_max_depth:
        for n_estimators in list_n_estimators:
            yield max_depth, n_estimators


def run_rf_classification_recall(X_train, X_test, y_train, y_test, max_depth, n_estimators, seed):
    """
    Score is species level recall.
    """

    rf = RandomForestClassifier(max_depth=max_depth,
                                n_estimators=n_estimators,
                                random_state=seed)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    score = recall_score(y_test, y_pred, average='weighted')
    return score


def grid_search_multiclass_rf(seq_file,
                              taxid_file,
                              output_dir,
                              pattern,
                              list_sample_length,
                              list_coverage,
                              list_k,
                              list_max_depth,
                              list_n_estimators,
                              seed,
                              grid_search_file,
                              fields,
                              experiment,
                              score_type):
    # set up grid search results file
    append_results_to_file(grid_search_file, fields=fields)

    # calculate number of combinations
    n_combinations = calc_number_combinations(list_sample_length,
                                              list_coverage,
                                              list_k,
                                              list_max_depth,
                                              list_n_estimators)

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
        for max_depth, n_estimators in hyperparameter_generator(list_max_depth, list_n_estimators):
            print(max_depth, n_estimators)

            # train and score model
            score = run_rf_classification_recall(X_train, X_test, y_train, y_test, max_depth, n_estimators,
                                                 seed)
            count += 1

            # output results to file
            row = [experiment, 'multiclass', 'Random Forest', X_train.shape, sample_length, coverage, k,
                   max_depth, n_estimators, score, score_type]
            append_results_to_file(grid_search_file, rows=row)

        print('Percent complete: {}'.format(count / n_combinations * 100))  # display progress


def main():
    # parameters
    seq_file = '/Users/ryanqnelson/GitHub/C-A-L-C-I-F-E-R/CMU-02620-Metagenomics/data/train_small-db_toy-5000.fasta'
    taxid_file = '/Users/ryanqnelson/GitHub/C-A-L-C-I-F-E-R/CMU-02620-Metagenomics/data/train_small-db_toy-5000.taxid'
    output_dir = '/Users/ryanqnelson/GitHub/C-A-L-C-I-F-E-R/CMU-02620-Metagenomics/data/sampling/sampling-toy-5000'
    pattern = 'fragments*.npy'
    seed = 42
    date_time = datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')
    data_dir = '/Users/ryanqnelson/GitHub/C-A-L-C-I-F-E-R/CMU-02620-Metagenomics/'
    grid_search_file = 'data/gridsearch-5000/results-5000-rf.{}.csv'.format(date_time)
    fields = ['experiment',
              'category',
              'classifier',
              'training shape',
              'sample_length',
              'coverage',
              'k',
              'max_depth',
              'n_estimators',
              'score',
              'score type']

    experiment = '11.02'
    score_type = 'species_recall'

    # combinations to try
    list_sample_length = [100, 200, 400] * 5
    list_coverage = [0.1, 1, 10, 100, 200, 400]
    list_k = [1, 2, 4, 6, 8, 10, 12]
    list_max_depth = [30]
    list_n_estimators = [50]

    grid_search_multiclass_rf(seq_file,
                              taxid_file,
                              output_dir,
                              pattern,
                              list_sample_length,
                              list_coverage,
                              list_k,
                              list_max_depth,
                              list_n_estimators,
                              seed,
                              grid_search_file,
                              fields,
                              experiment,
                              score_type)


if __name__ == "__main__":
    main()
