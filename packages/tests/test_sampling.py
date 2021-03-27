from Bio.Seq import Seq
from packages.metagenomics import sampling
import pytest
import numpy as np
import io
import os


def test__calc_number_fragments__whole_number():
    coverage = 0.1
    sample_length = 200
    sequence_length = 2000
    expected = 1.0
    actual = sampling._calc_number_fragments(sequence_length, coverage, sample_length)
    assert actual == expected


def test__calc_number_fragments__above_fifty_percent():
    coverage = 0.051
    sample_length = 200
    sequence_length = 2000
    expected = 1.0
    actual = sampling._calc_number_fragments(sequence_length, coverage, sample_length)
    assert actual == expected


def test__calc_number_fragments__below_fifty_percent():
    coverage = 0.04
    sample_length = 200
    sequence_length = 2000
    expected = 1.0
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


def test__draw_fragments__valid_sequence():
    seq = Seq("actgCtgatgtctactgtac")  # length of 20
    sample_length = 5
    n_frag = 4

    actual = sampling._draw_fragments(seq, sample_length, n_frag)

    # check number of fragments
    assert len(actual) == n_frag

    # check that all fragments are lowercase and contain only a,c,t,g
    allowed = ['a', 'c', 't', 'g']
    for frag in actual.tolist():
        assert all(c in allowed for c in frag.decode('utf-8'))


def test__draw_fragments__invalid_sequence():
    seq = Seq("actgCtgatUtctactgtac")  # length of 20
    sample_length = 5
    n_frag = 4

    actual = sampling._draw_fragments(seq, sample_length, n_frag)

    # check number of fragments
    assert len(actual) == n_frag

    for frag in actual.tolist():
        assert 'u' not in frag.decode('utf-8')


def test__draw_fragments__infinite_loop():
    seq = Seq("aUtgCUgatUtctUctgUac")  # no valid fragments
    sample_length = 5
    n_frag = 4
    with pytest.raises(ValueError):
        sampling._draw_fragments(seq, sample_length, n_frag)


def test__build_fragment_array():
    seq = Seq("actgCtgatgtctactgtac")  # length of 20
    sample_length = 5
    coverage = 1

    actual = sampling._build_fragment_array(seq, sample_length, coverage)

    # check number of fragments
    assert len(actual) == 4

    # check that all fragments are lowercase and contain only a,c,t,g
    allowed = ['a', 'c', 't', 'g']
    for frag in actual.tolist():
        assert all(c in allowed for c in frag.decode('utf-8'))


def test__build_fragment_array__random_seed():
    seq = Seq("actgCtgatgtctactgtac")  # length of 20
    sample_length = 5
    coverage = 1
    seed = 42

    actual = sampling._build_fragment_array(seq, sample_length, coverage, seed)

    # check number of fragments
    assert len(actual) == 4

    # check that all fragments are lowercase and contain only a,c,t,g
    allowed = ['a', 'c', 't', 'g']
    for frag in actual.tolist():
        assert all(c in allowed for c in frag.decode('utf-8'))


def test__build_fragment_array__seq_too_short():
    seq = Seq("act")  # length of 20
    sample_length = 5
    coverage = 1

    actual = sampling._build_fragment_array(seq, sample_length, coverage)

    # check number of fragments
    assert len(actual) == 0


def test__build_taxid_array():
    taxid = 'NC_013451'
    n_frag = 2
    expected = np.array([b'NC_013451', b'NC_013451'])
    actual = sampling._build_taxid_array(n_frag, taxid)
    np.testing.assert_array_equal(actual, expected)


def test__combine_fragments_and_taxids():
    fragments = np.array([b'atcg', b'gtcc'])
    taxids = np.array([b'NC_013451', b'NC_013451'])

    expected = np.array([[b'atcg', b'NC_013451'],
                         [b'gtcc', b'NC_013451']])
    actual = sampling._combine_fragments_and_taxids(fragments, taxids)
    np.testing.assert_array_equal(actual, expected)


def test__build_fragment_taxid_array():
    taxid = 'NC_013451'
    seq = Seq("actgCtgatgtctactgtac")  # length of 20
    sample_length = 5
    coverage = 1
    seed = 42

    actual = sampling._build_fragment_taxid_array(taxid, seq, sample_length, coverage, seed)

    # check number of fragments
    assert len(actual) == 4
    assert actual.shape == (4, 2)
    assert actual[0][1] == b'NC_013451'


def test__write_fragments(tmp_path):
    # temp directory
    d = tmp_path / '_write_fragments'
    d.mkdir()

    data = np.array([1, 2, 3])
    output_dir = d
    i = 1

    sampling._write_fragments(data, output_dir, i)

    # verify that file was generated
    expected_file = str(d) + '/fragments-00001.npy'
    assert os.path.isfile(expected_file)

    # verify that file contains the expected data
    actual = np.load(expected_file)
    expected = data
    np.testing.assert_array_equal(actual, expected)


def test__create_fragment_directory__directory_exists(tmp_path):
    # temp directory
    d = tmp_path / "sampling"
    d.mkdir()

    with pytest.raises(ValueError):
        sampling._create_fragment_directory(d)


def test__create_fragment_directory__directory_does_not_exist(tmp_path):
    # temp directory
    d = tmp_path / "sampling"

    sampling._create_fragment_directory(d)
    assert os.path.isdir(d)


def test__read_taxid_data__single_row(tmp_path):
    # create mockup files
    d = tmp_path  # use temp directory

    # taxid file
    taxid_file = d / 'tmp.taxid'
    taxid_contents = '1280'
    with open(taxid_file, 'w') as output_handle:
        output_handle.write(taxid_contents)

    actual = sampling._read_taxid_data(taxid_file)
    expected = np.array(['1280'])
    np.testing.assert_array_equal(actual, expected)


def test__read_taxid_data__multiple_rows(tmp_path):
    # create mockup files
    d = tmp_path  # use temp directory

    # taxid file
    taxid_file = d / 'tmp.taxid'
    taxid_contents = '1280\n1234'
    with open(taxid_file, 'w') as output_handle:
        output_handle.write(taxid_contents)

    actual = sampling._read_taxid_data(taxid_file)
    expected = np.array(['1280', '1234'])
    np.testing.assert_array_equal(actual, expected)


def test_generate_fragment_data__one_sequence(tmp_path):
    # create mockup files
    d = tmp_path  # use temp directory

    # seq file
    seq_file = d / 'tmp.seq'
    seq_contents = '>NC_013451\nagcaagcaccaacagcaatacatatagcctaaaggttccatgtccaaaaggaaattggaa'
    with open(seq_file, 'w') as output_handle:
        output_handle.write(seq_contents)

    # taxid file
    taxid_file = d / 'tmp.taxid'
    taxid_contents = '1280'
    with open(taxid_file, 'w') as output_handle:
        output_handle.write(taxid_contents)

    # other parameters
    output_dir = d / "sampling"
    sample_length = 5
    coverage = 1

    # run function
    sampling.generate_fragment_data(seq_file, taxid_file, output_dir, sample_length, coverage)

    # read in written file
    expected_file = output_dir / 'fragments-00000.npy'
    actual = np.load(expected_file)
    assert actual.shape == (12, 2)
    assert actual[0][1] == b'1280'


def test_generate_fragment_data__multiple_sequences(tmp_path):
    # create mockup files
    d = tmp_path  # use temp directory

    # seq file
    seq_file = d / 'tmp.seq'
    seq_contents = '>NC_013451\nagcaagcaccaacagcaatacatatagcctaaaggttccatgtccaaaaggaaattggaa\n' \
                   + '>NC_006375\naattcctagtttggcgacccggaacacgtgagttaatcttgaatattcgtatttactaga'

    with open(seq_file, 'w') as output_handle:
        output_handle.write(seq_contents)

    # taxid file
    taxid_file = d / 'tmp.taxid'
    taxid_contents = '1280\n88411'
    with open(taxid_file, 'w') as output_handle:
        output_handle.write(taxid_contents)

    # other parameters
    output_dir = d / "sampling"
    sample_length = 5
    coverage = 1

    # run function
    sampling.generate_fragment_data(seq_file, taxid_file, output_dir, sample_length, coverage)

    # read in written file
    expected_file1 = output_dir / 'fragments-00000.npy'
    actual = np.load(expected_file1)
    assert actual.shape == (12, 2)
    assert actual[0][1] == b'1280'

    expected_file2 = output_dir / 'fragments-00001.npy'
    actual = np.load(expected_file2)
    assert actual.shape == (12, 2)
    assert actual[0][1] == b'88411'


def test_read_fragments__one_file(tmp_path):
    # create mockup files
    d = tmp_path  # use temp directory

    f1 = np.array([[b'atcg', b'NC_013451'],
                   [b'gtcc', b'NC_013451']])

    output_file = d / 'fragment-00001.npy'
    with open(output_file, 'wb') as f:
        np.save(f, f1)

    actual = sampling.read_fragments(str(d), 'fragment*.npy')
    np.testing.assert_array_equal(actual, f1)


def test_read_fragments__multiple_files(tmp_path):
    # create mockup files
    d = tmp_path  # use temp directory

    f1 = np.array([[b'atcg', b'10312'],
                   [b'gtcc', b'10312']])

    f2 = np.array([[b'gggg', b'55921'],
                   [b'aaaa', b'55921']])

    output_file1 = d / 'fragment-00001.npy'
    with open(output_file1, 'wb') as f:
        np.save(f, f1)

    output_file2 = d / 'fragment-00002.npy'
    with open(output_file2, 'wb') as f:
        np.save(f, f2)

    expected = np.array([[b'atcg', b'10312'],
                         [b'gtcc', b'10312'],
                         [b'gggg', b'55921'],
                         [b'aaaa', b'55921']])

    actual = sampling.read_fragments(str(d), 'fragment*.npy')
    np.testing.assert_array_equal(actual, expected)
