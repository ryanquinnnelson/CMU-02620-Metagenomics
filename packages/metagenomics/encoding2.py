"""
Defines encoding functionality for metagenomics data.
Todo - Redesign splitting to instead slice k character array columns and use np.char.add() to form k-mers.
     (This should be more efficient.)

    a = np.array([  [b'a'],
                    [b'c']])

    b = np.array([  [b'b'],
                    [b'd']])

    np.char.add(a,b)
    >>> array([[b'ab'],
    >>>        [b'cd']], dtype='|S2')
"""
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder


# tested
def _get_kmer_start(k, i):
    """
    Determines the start position for the ith k-mer.

    :param k: int, length of kmer
    :param i: int, ith kmer
    :return: int, start position
    """

    return k * i


# tested
def _concatenate_cols(cols):
    """
    Concatenates the contents of all columns together into a single column.

    :param cols: n x c array, where c is the number of columns to concatenate
    :return: n x 1 array, concatenated contents
    """
    combined = cols[:, 0]

    # for each pair of columns
    n_cols = cols.shape[1]
    for i in range(n_cols - 1):
        combined = np.char.add(combined, cols[:, i + 1])

    return combined.reshape(-1, 1)


# tested
def _calculate_number_kmers(fragments, k):
    """
    Calculates the number of whole kmers which can be formed from the fragments.

    :param fragments: n x (L+1) array
    :param k: int, kmer size
    :return: int, number of kmers
    """
    n_fragment_columns = fragments.shape[1] - 1
    return math.floor(n_fragment_columns / k)


# tested
def _get_list_extra_columns(fragments, n_kmers):
    n_cols_fragments = fragments.shape[1]
    n_extra = n_cols_fragments - n_kmers - 1  # last (taxid) column should remain
    return [i for i in range(n_kmers, n_kmers + n_extra)]


# tested
def _group_kmers(fragments, k):
    """
    Groups kmers in place and returns array with kmers and taxids.
    Removes partial kmers.

    :param fragments: n x (L+1) array, where n is the number of fragments and L is the sample length
    :param k: int, length of kmer
    :return: n x (n_kmer + 1), where n_kmer is the number of whole kmers which can be formed from the sample length
    """
    # build kmers
    n_kmers = _calculate_number_kmers(fragments, k)
    for i in range(n_kmers):
        # grab columns for current kmer
        start_idx = _get_kmer_start(k, i)
        one_after_end = start_idx + k
        kmer_cols = fragments[:, start_idx:one_after_end]

        # concatenate kmer columns
        combined = _concatenate_cols(kmer_cols)

        # replace ith column in fragments with concatenated values
        # ith column now represents ith kmer
        fragments[:, i] = combined[:, 0]

    # delete extra columns between kmer columns and last (taxid) column
    # these columns have been concatenated into kmers and are no longer needed
    extra_col_idx = _get_list_extra_columns(fragments, n_kmers)
    grouped = np.delete(fragments, extra_col_idx, axis=1)
    return grouped


# tested
def encode_fragment_dataset(fragments, k):
    """
    Converts fragments into k-mers and encodes the data using one-hot encoding.

    :param fragments: fragment to be split
    :param k: size of elements fragment should be split into
    :return: sparse matrix
    """

    # generate k_mers
    data = _group_kmers(fragments, k).astype('str')  # convert binary data to string

    # split data into kmers and taxid
    X = np.delete(data, -1, axis=1)
    y = data[:, -1]

    # encode data using one-hot encoding
    X_enc = OneHotEncoder().fit_transform(X)

    return X_enc, y
