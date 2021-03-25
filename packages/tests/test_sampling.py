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
    expected = np.chararray((4, 5))
    expected[:] = '!'
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
    expected = np.array([b'm', b'k', b'q', b'h'], dtype='|S1')
    actual = sampling.convert_frag_seq(seq)
    np.testing.assert_array_equal(actual, expected)


def test_fragment_is_valid__success():
    frag = np.array([b'a', b'a', b'c', b't', b'g'], dtype='|S1')
    assert sampling.fragment_is_valid(frag)


def test_fragment_is_valid__failure():
    frag = np.array([b'a', b'a', b'c', b't', b'h'], dtype='|S1')
    assert sampling.fragment_is_valid(frag) is False


def test_draw_fragments__valid_sequence():
    record = SeqRecord(
        Seq("actgCtgatGtctactgtac"),  # length of 20
        id="YP_025292.1")
    sample_length = 5
    coverage = 1

    expected_n_frag = 4

    actual = sampling.draw_fragments(record, sample_length, coverage)

    # check number of fragments
    assert len(actual) == expected_n_frag

    for frag in actual.tolist():
        # check that all fragments are lowercase
        allowed = [b'a', b'c', b't', b'g']
        assert all(c in allowed for c in frag)


def test_draw_fragments__invalid_sequence():
    record = SeqRecord(
        Seq("actgCtgatUtctactgtac"),  # length of 20
        id="YP_025292.1")
    sample_length = 5
    coverage = 1

    expected_n_frag = 4

    actual = sampling.draw_fragments(record, sample_length, coverage)

    # check number of fragments
    assert len(actual) == expected_n_frag

    for frag in actual.tolist():
        assert b'u' in frag
