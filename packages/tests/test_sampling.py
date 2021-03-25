from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from packages.metagenomics import sampling
import numpy as np


def test_split_record():
    record = SeqRecord(
        Seq("MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"),
        id="YP_025292.1",
        name="HokC",
        description="toxic membrane protein, small")

    expected_id = "YP_025292.1"
    expected_seq = Seq("MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF")
    expected_seq_length = 44

    actual_id, actual_seq, actual_seq_length = sampling.split_record(record)
    assert actual_id == expected_id
    assert actual_seq == expected_seq
    assert actual_seq_length == expected_seq_length


def test_calc_number_fragments__whole_number():
    coverage = 0.1
    sample_length = 200
    sequence_length = 2000
    expected = 1.0
    actual = sampling.calc_number_fragments(sequence_length, coverage, sample_length)
    assert actual == expected


def test_calc_number_fragments__rounding_up():
    coverage = 0.051
    sample_length = 200
    sequence_length = 2000
    expected = 1.0
    actual = sampling.calc_number_fragments(sequence_length, coverage, sample_length)
    assert actual == expected


def test_calc_number_fragments__rounding_down():
    coverage = 0.05
    sample_length = 200
    sequence_length = 2000
    expected = 0.0
    actual = sampling.calc_number_fragments(sequence_length, coverage, sample_length)
    assert actual == expected


def test_create_chararray():
    n_frag = 4
    sample_length = 5
    expected = np.chararray((4, 6))
    expected[:] = '-'
    actual = sampling.create_chararray(n_frag, sample_length)
    np.testing.assert_array_equal(actual, expected)


def test_get_random_position():
    seq_length = 7
    sample_length = 3

    for i in range(1000):
        actual_start_pos, actual_end_pos = sampling.get_random_position(seq_length, sample_length)

        assert 0 <= actual_start_pos <= 4
        assert 3 <= actual_end_pos <= 7


def convert_frag_seq():
    seq = Seq("MKQH")
    expected = np.array(['m', 'k', 'q', 'h', '-'])
    actual = sampling.convert_frag_seq(seq)
    np.testing.assert_array_equal(actual, expected)


def test_fragment_is_valid__success():
    frag = np.array(['a', 'a', 'c', 't', 'g'])
    assert sampling.fragment_is_valid(frag)


def test_fragment_is_valid__failure():
    frag = np.array(['a', 'a', 'c', 't', 'h'])
    assert sampling.fragment_is_valid(frag) is False


def test_update_frag_array__valid():
    frag = np.array(['a', 'c', 't', 'g', '-'])
    charar = np.chararray((3, 5))
    charar[:] = '-'
    is_valid = True
    i = 1
    expected = np.array([['-', '-', '-', '-', '-'],
                         ['a', 'c', 't', 'g', 'v'],
                         ['-', '-', '-', '-', '-']], dtype='|S1')
    actual = sampling.update_frag_array(frag, charar, is_valid, i)
    np.testing.assert_array_equal(actual, expected)


def test_update_frag_array__invalid():
    frag = np.array(['a', 'c', 't', 'g', '-'])
    charar = np.chararray((3, 5))
    charar[:] = '-'
    is_valid = False
    i = 1
    expected = np.array([['-', '-', '-', '-', '-'],
                         ['a', 'c', 't', 'g', 'i'],
                         ['-', '-', '-', '-', '-']], dtype='|S1')
    actual = sampling.update_frag_array(frag, charar, is_valid, i)
    np.testing.assert_array_equal(actual, expected)


def test_get_valid_rows__one_row():
    charar = np.array([['-', '-', '-', '-', '-'],
                       ['a', 'c', 't', 'g', 'i'],
                       ['a', 'g', 't', 'g', 'v']], dtype='|S1')
    expected = np.array([[2]])
    actual = sampling.get_valid_rows(charar)
    np.testing.assert_array_equal(actual, expected)


def test_get_valid_rows__muliple_rows():
    charar = np.array([['-', '-', '-', '-', 'i'],
                       ['a', 'c', 't', 'g', 'v'],
                       ['a', 'g', 't', 'g', 'v']], dtype='|S1')
    expected = np.array([[1, 2]])
    actual = sampling.get_valid_rows(charar)
    np.testing.assert_array_equal(actual, expected)


def test_remove_invalid_frags__one_row_left():
    charar = np.array([['-', '-', '-', '-', 'i'],
                       ['a', 'c', 't', 'g', 'i'],
                       ['a', 'g', 't', 'g', 'v']], dtype='|S1')
    expected = np.array([['a', 'g', 't', 'g']], dtype='|S1')
    actual = sampling.remove_invalid_frags(charar)
    np.testing.assert_array_equal(actual, expected)


def test_remove_invalid_frags__multiple_rows_left():
    charar = np.array([['-', '-', '-', '-', 'i'],
                       ['a', 'c', 't', 'g', 'v'],
                       ['a', 'g', 't', 'g', 'v']], dtype='|S1')
    expected = np.array([['a', 'c', 't', 'g'],
                         ['a', 'g', 't', 'g']], dtype='|S1')
    actual = sampling.remove_invalid_frags(charar)
    np.testing.assert_array_equal(actual, expected)


def test_draw_fragments():
    record = SeqRecord(
        Seq("actgCtgatGtctactgtac"),  # length of 20
        id="YP_025292.1")
    sample_length = 5
    coverage = 1

    expected_n_frag = 4

    actual = sampling.draw_fragments(record, sample_length, coverage)

    # check number of fragments
    assert len(actual) == expected_n_frag