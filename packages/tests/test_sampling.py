from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from packages.metagenomics import sampling
import numpy as np
import pytest


# def test_split_record():
#     record = SeqRecord(
#         Seq("MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"),
#         id="YP_025292.1",
#         name="HokC",
#         description="toxic membrane protein, small")
#
#     expected_id = "YP_025292.1"
#     expected_seq = Seq("MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF")
#     expected_seq_length = 44
#
#     actual_id, actual_seq, actual_seq_length = sampling.split_record(record)
#     assert actual_id == expected_id
#     assert actual_seq == expected_seq
#     assert actual_seq_length == expected_seq_length
#
#
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


def test_get_random_position():
    seq_length = 7
    sample_length = 3

    for i in range(1000):
        actual_start_pos = sampling.get_random_position(seq_length, sample_length)

        assert 0 <= actual_start_pos <= 4


def test_fragment_is_valid__success():
    frag = 'aactg'
    assert sampling.fragment_is_valid(frag)


def test_fragment_is_valid__failure():
    frag = 'aacth'
    assert sampling.fragment_is_valid(frag) is False


def test_draw_fragment():
    seq = Seq("actgCtgatgtctactgtac")  # length of 20
    sample_length = 5
    actual = sampling.draw_fragment(seq, sample_length)
    assert len(actual) == 5
    assert actual in seq.lower()


def test_draw_fragments__valid_sequence():
    seq = Seq("actgCtgatgtctactgtac")  # length of 20
    sample_length = 5
    coverage = 1

    expected_n_frag = 4

    actual = sampling.draw_fragments(seq, sample_length, coverage)

    # check number of fragments
    assert len(actual) == expected_n_frag

    # check that all fragments are lowercase and contain only a,c,t,g
    allowed = ['a', 'c', 't', 'g']
    for frag in actual.tolist():
        assert all(c in allowed for c in frag.decode('utf-8'))


def test_draw_fragments__invalid_sequence():
    seq = Seq("actgCtgatUtctactgtac")  # length of 20
    sample_length = 5
    coverage = 1

    expected_n_frag = 4

    actual = sampling.draw_fragments(seq, sample_length, coverage)

    # check number of fragments
    assert len(actual) == expected_n_frag

    for frag in actual.tolist():
        assert 'u' not in frag.decode('utf-8')


def test_draw_fragments__infinite_loop():
    seq = Seq("aUtgCUgatUtctUctgUac")  # no valid fragments
    sample_length = 5
    coverage = 1

    with pytest.raises(ValueError):
        sampling.draw_fragments(seq, sample_length, coverage)


def test_build_taxid_array():
    taxid = 'NC_013451'
    n_frag = 2
    expected = np.array([b'NC_013451', b'NC_013451'])
    actual = sampling.build_taxid_array(n_frag, taxid)
    np.testing.assert_array_equal(actual, expected)


def test_build_output_rows():
    fragments = np.array([b'atcg', b'gtcc'])
    taxid = 'NC_013451'
    expected = np.array([[b'NC_013451', b'atcg'],
                         [b'NC_013451', b'gtcc']])
    actual = sampling.build_output_rows(fragments, taxid)
    np.testing.assert_array_equal(actual, expected)
