import numpy as np
from packages.metagenomics import encoding


def test__split_into_kmers__partial_fragment():
    fragment = b'atcgatcc'
    k = 3
    expected = np.array([b'atc', b'gat'])
    actual = encoding._split_into_kmers(fragment, k)
    np.testing.assert_array_equal(actual, expected)


def test__split_into_kmers__exact_subdivision():
    fragment = b'atcgatccc'
    k = 3
    expected = np.array([b'atc', b'gat', b'ccc'])
    actual = encoding._split_into_kmers(fragment, k)
    np.testing.assert_array_equal(actual, expected)


def test__generate_kmers():
    fragments = np.array([b'atcggaagtc', b'gtccaaatcg'])
    k = 4
    expected = np.array([[b'atcg', b'gaag'],
                         [b'gtcc', b'aaat']])
    actual = encoding._generate_kmers(fragments, k)
    np.testing.assert_array_equal(actual, expected)


def test_encode_fragment_dataset():
    fragments = np.array([b'atcggaagtc', b'gtccaaatcg'])
    k = 4
    expected = np.array([[1., 0., 0., 1.],
                         [0., 1., 1., 0.]])
    actual = encoding.encode_fragment_dataset(fragments, k)
    np.testing.assert_array_equal(actual.toarray(), expected)



