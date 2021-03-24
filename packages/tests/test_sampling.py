from Bio.Seq import Seq
from packages.metagenomics import sampling


def test_sequence_is_valid__success():
    seq = Seq("atcg")
    assert sampling.sequence_is_valid(seq)


def test_sequence_is_valid__failure():
    seq = Seq("atcgh")
    assert sampling.sequence_is_valid(seq) is False


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
    expected = 0.0
    actual = sampling.calc_number_fragments(sequence_length, coverage, sample_length)
    assert actual == expected


def test_calc_number_fragments__rounding_down():
    coverage = 0.05
    sample_length = 200
    sequence_length = 2000
    expected = 1.0
    actual = sampling.calc_number_fragments(sequence_length, coverage, sample_length)
    assert actual == expected
