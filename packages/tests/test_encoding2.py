import numpy as np
from packages.metagenomics import encoding2


def test__get_kmer_start__first():
    k = 3
    i = 0
    actual = encoding2._get_kmer_start(k, i)
    assert actual == 0


def test__get_kmer_start__second():
    k = 3
    i = 1
    actual = encoding2._get_kmer_start(k, i)
    assert actual == 3


def test__concatenate_cols__three_columns():
    cols = np.array([[b'g', b'a', b't'],
                     [b'g', b'c', b't'],
                     [b't', b'a', b'c'],
                     [b'c', b't', b'g']])

    expected = np.array([[b'gat'],
                         [b'gct'],
                         [b'tac'],
                         [b'ctg']])

    actual = encoding2._concatenate_cols(cols)
    np.testing.assert_array_equal(actual, expected)


def test__concatenate_cols__two_columns():
    cols = np.array([[b'g', b'a'],
                     [b'g', b'c'],
                     [b't', b'a'],
                     [b'c', b't']])

    expected = np.array([[b'ga'],
                         [b'gc'],
                         [b'ta'],
                         [b'ct']])

    actual = encoding2._concatenate_cols(cols)
    np.testing.assert_array_equal(actual, expected)


def test__calculate_number_kmers():
    fragments = np.array([[1, 2, 3, 4, 5],
                          [1, 2, 3, 4, 5]])
    k = 2
    actual = encoding2._calculate_number_kmers(fragments, k)
    assert actual == 2


def test__get_list_extra_columns():
    fragments = np.array([[1, 2, 3, 4, 5],
                          [1, 2, 3, 4, 5]])
    k = 2
    expected = [2, 3]
    actual = encoding2._get_list_extra_columns(fragments, k)
    assert actual == expected


def test__group_kmers__partial_kmer():
    fragments = np.array([[b'g', b'a', b't', b'g', b't', b'128221'],
                          [b'g', b'c', b't', b'g', b'a', b'128221'],
                          [b't', b'a', b'c', b't', b'g', b'128221'],
                          [b'c', b't', b'g', b't', b'a', b'128221']])

    k = 2

    X_expected = np.array([[b'ga', b'tg'],
                           [b'gc', b'tg'],
                           [b'ta', b'ct'],
                           [b'ct', b'gt']])
    y_expected = np.array([b'128221', b'128221', b'128221', b'128221'])

    X_actual, y_actual = encoding2._group_kmers(fragments, k)
    np.testing.assert_array_equal(X_actual, X_expected)
    np.testing.assert_array_equal(y_actual, y_expected)


def test__group_kmers__full_kmers():
    fragments = np.array([[b'g', b'a', b't', b'g', b't', b'a', b'128221'],
                          [b'g', b'c', b't', b'g', b'a', b'a', b'128221'],
                          [b't', b'a', b'c', b't', b'g', b'a', b'128221'],
                          [b'c', b't', b'g', b't', b'a', b'a', b'128221']])

    k = 3

    X_expected = np.array([[b'gat', b'gta'],
                           [b'gct', b'gaa'],
                           [b'tac', b'tga'],
                           [b'ctg', b'taa']])
    y_expected = np.array([b'128221', b'128221', b'128221', b'128221'])
    X_actual, y_actual = encoding2._group_kmers(fragments, k)
    np.testing.assert_array_equal(X_actual, X_expected)
    np.testing.assert_array_equal(y_actual, y_expected)


def test_encode_fragment_dataset():
    fragments = np.array([[b'g', b'a', b't', b'g', b't', b'a', b'128221'],
                          [b'g', b'c', b't', b'g', b'a', b'a', b'128221'],
                          [b't', b'a', b'c', b't', b'g', b'a', b'128221'],
                          [b'c', b't', b'g', b't', b'a', b'a', b'128221']])

    k = 3

    X_expected = np.array([[0., 1., 0., 0., 0., 1., 0., 0.],
                           [0., 0., 1., 0., 1., 0., 0., 0.],
                           [0., 0., 0., 1., 0., 0., 0., 1.],
                           [1., 0., 0., 0., 0., 0., 1., 0.]])
    y_expected = np.array(['128221', '128221', '128221', '128221'])

    X_actual, y_actual = encoding2.encode_fragment_dataset(fragments, k)
    np.testing.assert_array_equal(X_actual.toarray(), X_expected)
    np.testing.assert_array_equal(y_actual, y_expected)
