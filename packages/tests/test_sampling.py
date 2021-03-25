from Bio.Seq import Seq
from packages.metagenomics import sampling
import pytest


def test__calc_number_fragments__whole_number():
    coverage = 0.1
    sample_length = 200
    sequence_length = 2000
    expected = 1.0
    actual = sampling._calc_number_fragments(sequence_length, coverage, sample_length)
    assert actual == expected


def test__calc_number_fragments__rounding_up():
    coverage = 0.051
    sample_length = 200
    sequence_length = 2000
    expected = 1.0
    actual = sampling._calc_number_fragments(sequence_length, coverage, sample_length)
    assert actual == expected


def test__calc_number_fragments__rounding_down():
    coverage = 0.05
    sample_length = 200
    sequence_length = 2000
    expected = 0.0
    actual = sampling._calc_number_fragments(sequence_length, coverage, sample_length)
    assert actual == expected


def test__get_random_position():
    seq_length = 7
    sample_length = 3

    for i in range(1000):
        actual_start_pos = sampling._get_random_position(seq_length, sample_length)

        assert 0 <= actual_start_pos <= 4


def test__fragment_is_valid__success():
    frag = 'aactg'
    assert sampling._fragment_is_valid(frag)


def test__fragment_is_valid__failure():
    frag = 'aacth'
    assert sampling._fragment_is_valid(frag) is False


def test__draw_fragment():
    seq = Seq("actgCtgatgtctactgtac")  # length of 20
    sample_length = 5
    actual = sampling._draw_fragment(seq, sample_length)
    assert len(actual) == 5
    assert actual in seq.lower()


def test__build_fragment_array__valid_sequence():
    seq = Seq("actgCtgatgtctactgtac")  # length of 20
    sample_length = 5
    n_frag = 4

    actual = sampling._build_fragment_array(seq, sample_length, n_frag)

    # check number of fragments
    assert len(actual) == n_frag

    # check that all fragments are lowercase and contain only a,c,t,g
    allowed = ['a', 'c', 't', 'g']
    for frag in actual.tolist():
        assert all(c in allowed for c in frag.decode('utf-8'))


def test__build_fragment_array__invalid_sequence():
    seq = Seq("actgCtgatUtctactgtac")  # length of 20
    sample_length = 5
    n_frag = 4

    actual = sampling._build_fragment_array(seq, sample_length, n_frag)

    # check number of fragments
    assert len(actual) == n_frag

    for frag in actual.tolist():
        assert 'u' not in frag.decode('utf-8')


def test__build_fragment_array__infinite_loop():
    seq = Seq("aUtgCUgatUtctUctgUac")  # no valid fragments
    sample_length = 5
    n_frag = 4
    with pytest.raises(ValueError):
        sampling._build_fragment_array(seq, sample_length, n_frag)


def test_draw_fragments():
    seq = Seq("actgCtgatgtctactgtac")  # length of 20
    sample_length = 5
    coverage = 1
    seed = 42

    actual = sampling.draw_fragments(seq, sample_length, coverage, seed)

    # check number of fragments
    assert len(actual) == 4

    # check that all fragments are lowercase and contain only a,c,t,g
    allowed = ['a', 'c', 't', 'g']
    for frag in actual.tolist():
        assert all(c in allowed for c in frag.decode('utf-8'))


def test_draw_fragments__seq_too_short():
    seq = Seq("act")  # length of 20
    sample_length = 5
    coverage = 1
    seed = 0

    actual = sampling.draw_fragments(seq, sample_length, coverage, seed)

    # check number of fragments
    assert actual is None
