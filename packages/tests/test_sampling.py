from Bio.Seq import Seq
from packages.metagenomics import sampling
import pytest
import numpy as np
import io


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


def test__draw_fragments_for_sequence():
    seq = Seq("actgCtgatgtctactgtac")  # length of 20
    sample_length = 5
    coverage = 1

    actual = sampling._draw_fragments_for_sequence(seq, sample_length, coverage)

    # check number of fragments
    assert len(actual) == 4

    # check that all fragments are lowercase and contain only a,c,t,g
    allowed = ['a', 'c', 't', 'g']
    for frag in actual.tolist():
        assert all(c in allowed for c in frag.decode('utf-8'))


def test__draw_fragments_for_sequence__seq_too_short():
    seq = Seq("act")  # length of 20
    sample_length = 5
    coverage = 1

    actual = sampling._draw_fragments_for_sequence(seq, sample_length, coverage)

    # check number of fragments
    assert len(actual) == 0


def test__build_taxid_array():
    taxid = 'NC_013451'
    n_frag = 2
    expected = np.array([b'NC_013451', b'NC_013451'])
    actual = sampling._build_taxid_array(n_frag, taxid)
    np.testing.assert_array_equal(actual, expected)


def test__build_fragment_rows_for_sequence():
    fragments = np.array([b'atcg', b'gtcc'])
    taxid = 'NC_013451'
    expected = np.array([[b'NC_013451', b'atcg'],
                         [b'NC_013451', b'gtcc']])
    actual = sampling._build_fragment_rows_for_sequence(fragments, taxid)
    np.testing.assert_array_equal(actual, expected)


def test_draw_fragments(tmp_path):
    # generate temporary files
    # temp directory
    d = tmp_path / "sampling"
    d.mkdir()

    # temp files
    seq_file = d / 'tmp.seq'
    taxid_file = d / 'tmp.taxid'
    seq_contents = '>NC_013451\nagcaagcaccaacagcaatacatatagcctaaaggttccatgtccaaaaggaaattggaa'
    taxid_contents = '1280\n1280'
    with open(seq_file, 'w') as output_handle:
        output_handle.write(seq_contents)

    with open(taxid_file, 'w') as output_handle:
        output_handle.write(taxid_contents)

    output_dir = d
    sample_length = 5
    coverage = 1

    # run function
    sampling.draw_fragments(seq_file, taxid_file, output_dir, sample_length, coverage)

    # read in written file
    output_file = output_dir / 'fragments-00000.npy'
    print(output_file)
    fragments = np.load(output_file)
    assert fragments.shape == (12, 2)
    assert fragments[0][0] == b'1280'
