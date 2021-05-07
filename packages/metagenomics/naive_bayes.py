'''
Implementation of Naive Bayes classifier for metagenomics data generated from encoding2 module.
'''
from sklearn.metrics import recall_score
import numpy as np


def run_naive_bayes(X_train, X_test, y_train, y_test):
    '''
    Runs this implementation of Naive Bayes, returning the recall score for the predictions

    :param: X_train, our one-hot encoded training sequences
    :param: X_test, our one-hot encoded test sequences
    :param: y_train, the known taxids for the training sequences
    :param: y_test, the known taxids for the test sequences
    :return: the recall score for the predictions given by the "test_new_data" function
    '''

    taxid_probabilities = taxid_probability(y_train)
    sparse_list = split_sparsemtx_by_taxid(X_train, y_train)

    prediction_list = []
    for i in range(X_test.shape[0]):
        new_prediction = test_new_data(X_test[i, :], sparse_list, taxid_probabilities)
        prediction_list.append(max(new_prediction, key=new_prediction.get))


    score = recall_score(y_test, prediction_list, average='weighted')
    return score


def taxid_probability(taxids):
    '''
    Returns the probabilities of each taxid given an array of taxids.

    :param taxids: L x 1 array
    :return: dictionary, where keys are the taxids and values are the probabilities
    '''

    taxid_probability_dict = {}

    unique_taxids, taxid_counts = np.unique(taxids, return_counts=True)
    for i in range(len(unique_taxids)):
        taxid_probability_dict[unique_taxids[i]] = taxid_counts[i]/sum(taxid_counts)

    return taxid_probability_dict


def split_sparsemtx_by_taxid(matrix, taxids):
    '''
    Splits a sparse matrix into separate sparse matrices for each individual taxid present.

    :param matrix: a sparse matrix
    :param taxids: np.array of taxids
    :return: a list of sparse matrices, one for each taxid
    '''

    taxids_list, taxid_counts = np.unique(taxids, return_counts=True)
    sparse_mtx_list = []

    for each in taxids_list:
        index_list = np.array([i for i, val in enumerate(taxids) if val == each])
        sparse_mtx_list.append(matrix[index_list,:])

    return sparse_mtx_list


def test_new_data(encoded, sparse_list, taxid_probabilities):
    '''
    Calculates the probability that the encoded data belongs to each class

    :param encoded: a csr.matrix, corresponds to one row of the overall test matrix
    :param sparse_list:, a list of sparse matrices, each element corresponding to one taxid
    :param taxid_probabilities:, a dictionary, keys = taxids, values = probabilites
    :return: a dictionary where keys = taxids and values = probability of the test sequence belonging
        to that taxid
    '''

    final_probabilities = {}
    taxid_keys = list(taxid_probabilities.keys())

    encoded_to_array = encoded.toarray()

    for i in range(len(sparse_list)):
        col_sum = sparse_list[i].sum(axis=0)
        col_sum /= sparse_list[i].shape[0]

        prod = np.multiply(encoded_to_array, col_sum)
        # multiplying by the taxid probability
        total_prod = np.multiply(prod, taxid_probabilities[taxid_keys[i]])
        # print(total_prod)
        final_probabilities[taxid_keys[i]] = total_prod.sum().max()

    return final_probabilities
