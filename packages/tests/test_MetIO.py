from packages.metagenomics import MetIO
import numpy as np


def test_build_taxid_array():
    taxid = 'NC_013451'
    n_frag = 2
    expected = np.array([b'NC_013451', b'NC_013451'])
    actual = MetIO.build_taxid_array(n_frag, taxid)
    np.testing.assert_array_equal(actual, expected)


def test_build_output_rows():
    fragments = np.array([b'atcg', b'gtcc'])
    taxid = 'NC_013451'
    expected = np.array([[b'NC_013451', b'atcg'],
                         [b'NC_013451', b'gtcc']])
    actual = MetIO.build_output_rows(fragments, taxid)
    np.testing.assert_array_equal(actual, expected)
