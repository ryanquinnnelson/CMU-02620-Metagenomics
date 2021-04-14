import datetime

from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression

from packages.gridsearch.helper import append_results_to_file, build_fragments, encode_fragments, \
    calc_number_combinations, parameter_generator


def hyperparameter_generator_lr(list_penalty, list_multiclass, list_classweight, list_solver):
    """

    :param list_penalty:
    :param list_multiclass:
    :param list_classweight:
    :param list_solver:
    :return:
    """
    for penalty in list_penalty:
        for multiclass in list_multiclass:
            for classweight in list_classweight:
                for solver in list_solver:
                    yield penalty, multiclass, classweight, solver


def run_lr_classification_recall(X_train, X_test, y_train, y_test, penalty, multiclass, classweight, solver, seed):
    """
    Score is species level recall.
    Todo - add ability to Sets solver to 'saga' for l1 penalty. Uses default solver for l2 penalty. solver='saga'

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param penalty:
    :param multiclass:
    :param classweight:
    :param solver:
    :param seed:
    :return:
    """
    lr = LogisticRegression(penalty=penalty,
                            multi_class=multiclass,
                            class_weight=classweight,
                            solver=solver,
                            random_state=seed)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    score = recall_score(y_test, y_pred, average='weighted')
    return score


def grid_search_multiclass_lr(seq_file,
                              taxid_file,
                              output_dir,
                              pattern,
                              list_sample_length,
                              list_coverage,
                              list_k,
                              list_penalty,
                              list_multiclass,
                              list_classweight,
                              list_solver,
                              seed,
                              grid_search_file,
                              fields,
                              experiment,
                              score_type):
    """

    :param seq_file:
    :param taxid_file:
    :param output_dir:
    :param pattern:
    :param list_sample_length:
    :param list_coverage:
    :param list_k:
    :param list_penalty:
    :param list_multiclass:
    :param list_classweight:
    :param list_solver:
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
                                              list_penalty,
                                              list_multiclass,
                                              list_classweight,
                                              list_solver)

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
        for penalty, multiclass, classweight, solver in hyperparameter_generator_lr(list_penalty, list_multiclass,
                                                                                    list_classweight, list_solver):
            print(penalty, multiclass, classweight, solver)

            # train and score model
            score = run_lr_classification_recall(X_train, X_test, y_train, y_test, penalty, multiclass, classweight,
                                                 solver, seed)
            count += 1

            # output results to file
            row = [experiment, 'multiclass', 'Logistic Regression (sklearn)', X_train.shape, sample_length, coverage, k,
                   penalty, multiclass, classweight, solver, score, score_type]
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
    data_dir = '/Users/ryanqnelson/GitHub/C-A-L-C-I-F-E-R/CMU-02620-Metagenomics/'
    date_time = datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')
    grid_search_file = data_dir + 'data/gridsearch-3000/results-3000-lrpackage-l2.{}.csv'.format(date_time)
    fields = ['experiment',
              'category',
              'classifier',
              'training shape',
              'sample_length',
              'coverage',
              'k',
              'penalty',
              'multi_class',
              'class_weight',
              'solver',
              'score',
              'score type']
    experiment = '17.03'
    score_type = 'species_recall'

    # combinations to try
    list_sample_length = [100, 200, 400]
    list_coverage = [1, 10, 100, 200, 400]
    list_k = [1, 2, 4, 6, 8, 10, 12]
    list_penalty = ['l2']
    list_multiclass = ['auto']
    list_classweight = [None]
    list_solver = ['lbfgs']

    grid_search_multiclass_lr(seq_file,
                              taxid_file,
                              output_dir,
                              pattern,
                              list_sample_length,
                              list_coverage,
                              list_k,
                              list_penalty,
                              list_multiclass,
                              list_classweight,
                              list_solver,
                              seed,
                              grid_search_file,
                              fields,
                              experiment,
                              score_type)


if __name__ == "__main__":
    main()
