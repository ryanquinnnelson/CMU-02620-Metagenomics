import datetime
from sklearn.metrics import recall_score
from packages.gridsearch.helper import append_results_to_file, build_fragments, encode_fragments,\
    calc_number_combinations, parameter_generator
from packages.metagenomics.naive_bayes import run_naive_bayes

def grid_search_NB(seq_file,
                    taxid_file,
                    output_dir,
                    pattern,
                    list_sample_length,
                    list_coverage,
                    list_k,
                    seed,
                    grid_search_file,
                    fields,
                    experiment,
                    score_type):
    #
    append_results_to_file(grid_search_file, fields=fields)

    #
    n_combinations = calc_number_combinations(list_sample_length, list_coverage, list_k)

    #
    count = 0
    sample_len_prev = -1
    coverage_prev = -1

    for sample_len, coverage, k, in parameter_generator(list_sample_length, list_coverage, list_k):
        print(sample_len, coverage, k)

        if sample_len != sample_len_prev or coverage != coverage_prev:
            #
            build_fragments(seq_file, taxid_file, output_dir, sample_len, coverage, seed)
            sample_len_prev = sample_len
            coverage_prev = coverage

        X_train, X_test, y_train, y_test = encode_fragments(output_dir, pattern, k, seed)




        score = run_naive_bayes(X_train, X_test, y_train, y_test)

        count += 1

        row = [experiment, "Naive Bayes", X_train.shape, sample_len, coverage, k, score, score_type]
        append_results_to_file(grid_search_file, rows=row)

        print("Percent complete: {}".format(count/n_combinations * 100))


def main():
    seq_file = '/Users/pskim/Documents/ML Projects/CMU-02620-Metagenomics-main/data/train_small-db_toy-3000.fasta'
    taxid_file = '/Users/pskim/Documents/ML Projects/CMU-02620-Metagenomics-main/data/train_small-db_toy-3000.taxid'
    output_dir = '/Users/pskim/Documents/ML Projects/CMU-02620-Metagenomics-main/data/sampling/sampling-toy-3000'
    pattern = 'fragments*.npy'
    seed = None
    date_time = datetime.datetime.now().strftime('%Y.%m.%d.%H.%M')
    data_dir = '/Users/pskim/Documents/ML Projects/CMU-02620-Metagenomics-main/'
    grid_search_file = data_dir + 'data/gridsearch-3000/results-3000-naivebayes-single.{}.csv'.format(date_time)
    fields = ['experiment', 'classifier', 'training_shape', 'sample_length', 'coverage',
              'k', 'score', 'score_type']

    experiment = '5.01'
    score_type = 'species_recall'

    list_sample_length = [100, 200, 400]
    list_coverage = [1, 10, 100, 200, 400]
    list_k = [1, 2, 4, 6, 8, 10, 12]

    grid_search_NB(seq_file, taxid_file, output_dir, pattern, list_sample_length, list_coverage,
                    list_k, seed, grid_search_file, fields, experiment, score_type)


if __name__ == '__main__':
    main()
