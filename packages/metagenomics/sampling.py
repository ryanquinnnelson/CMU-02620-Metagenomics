"""
Defines sampling functionality for metagenomics data.
"""
import numpy as np
from Bio import SeqIO
from glob import glob
import os


# tested
def _calc_number_fragments(seq_length, coverage, sample_length):
    """
    Calculates the number of fragments to be randomly sampled from the given sequence in order to
    achieve desired coverage. Uses formula defined in Vervier et al. See https://arxiv.org/abs/1505.06915.

    :param seq_length: int, length of sequence to be sampled
    :param coverage: float, desired coverage
            (0.1 for 10% of bp coverage; 1 for 100% bp coverage; 10 for 10x bp coverage).
    :param sample_length: int, length of samples
    :return: int, number of fragments to sample
    """
    n_frag = seq_length * coverage / sample_length
    return round(n_frag)


# tested
def _get_random_position(seq_length, sample_length):
    """
    Selects a random start position for a sample, considering the length of the sequence and the length of the sample.

    :param seq_length: int, length of sequence to be sampled
    :param sample_length: int, length of samples
    :return: int, starting position defining subsequence of sample_length in sequence
    """
    return np.random.randint(0, seq_length - sample_length)


# tested
def _fragment_is_valid(frag):
    """
    Determines if fragment meets criteria required to be valid. Currently, the criteria is that all letters in the
    fragment are lowercase and encode DNA nucleotides i.e. {a,c,t,g}.

    :param frag: str, fragment selected from sequence
    :return: True if fragment is valid, false otherwise.
    """
    allowed = ['a', 'c', 't', 'g']
    return all(c in allowed for c in frag)


# tested
def _draw_fragment(seq, sample_length):
    """
    Draws one fragment sample at random from the given sequence.

    :param seq: Bio.Seq.Seq, sequence to be sampled
    :param sample_length: int, length of samples
    :return: str, lowercase string representing subsequence
    """
    # choose random subsequence
    seq_length = len(seq)
    start_pos = _get_random_position(seq_length, sample_length)

    # get fragment
    one_after_end = start_pos + sample_length
    frag_seq = seq[start_pos:one_after_end].lower()
    return str(frag_seq)


# tested
def _draw_fragments(seq, sample_length, n_frag):
    """
    Draws required number of valid fragments from sequence.
    Raises ValueError if too many invalid sequences are sampled in order to prevent an infinite loop in the case that
    the sequence does not contain valid subsequences of sample_length.

    :param seq: Bio.Seq.Seq, sequence to be sampled
    :param sample_length: int, length of samples
    :param n_frag: int, number of fragments to sample
    :return: n_frag x 1 array, valid fragments drawn from sample
    """
    fragments = np.chararray((n_frag,), itemsize=sample_length)  # scaffold for fragments

    # draw fragments
    n_valid = 0
    n_failures = 0
    while n_valid < n_frag:

        # draw random fragment
        frag = _draw_fragment(seq, sample_length)

        # determine whether to save or discard fragment
        if _fragment_is_valid(frag):
            fragments[n_valid] = frag  # save in array
            n_valid += 1
        else:
            n_failures += 1  # update breakout counter

        # determine if breakout is necessary
        if n_failures > n_frag * 10:
            raise ValueError(
                'Too many invalid fragments encountered. Sampling is stopped after {} attempts.'.format(n_failures))

    return fragments


# tested
def _build_fragment_array(seq, sample_length, coverage, seed=None):
    """
    Draws number of samples from sequence in order to achieve desired coverage and constructs an array of the results.
    Follows general sampling procedure laid out by Vervier et al. See https://arxiv.org/abs/1505.06915.

    :param seq: Bio.Seq.Seq, sequence to be sampled
    :param sample_length: int, length of samples
    :param coverage: float, desired coverage
            (0.1 for 10% of bp coverage; 1 for 100% bp coverage; 10 for 10x bp coverage).
    :param seed: int, random seed for reproducibility. Default is None.
    :return: n x 1 array, where n is the number of fragments drawn from sample in order to meet required coverage.
            Returns empty array if sequence length is less than sample length.
    """

    # initialize random seed if necessary
    if seed:
        np.random.seed(seed)

    # sample fragments if possible
    seq_length = len(seq)
    if seq_length >= sample_length:
        n_frag = _calc_number_fragments(seq_length, coverage, sample_length)
        fragments = _draw_fragments(seq, sample_length, n_frag)
    else:
        fragments = np.empty(0, )

    return fragments


# tested
def _build_taxid_array(n_frag, taxid):
    """
    Generates an array equal in length to the fragment array and fills it with the same taxid value for all cells.

    :param n_frag:  int, number of fragments to sample
    :param taxid: str, species for the sequence
    :return: n_frag x 1 array
    """
    taxid_length = len(taxid)
    taxids = np.chararray((n_frag,), itemsize=taxid_length)
    taxids[:] = taxid

    return taxids


# tested
def _combine_fragments_and_taxids(fragments, taxids):
    """
    Stacks fragments array and taxids array into a single matrix.

    :param fragments: n_frag x 1 array, sampled fragments for the sequence
    :param taxids: n_frag x 1 array, species for the sequence
    :return: n_frag x 2 matrix
    """
    return np.column_stack((fragments, taxids))


# tested
def _build_fragment_taxid_array(taxid, seq, sample_length, coverage, seed=None):
    """
    Builds dataset of fragments and the corresponding (identical) taxid for each fragment.

    :param taxid: str, species for the sequence
    :param seq: Bio.Seq.Seq, sequence to be sampled
    :param sample_length: int, length of samples
    :param coverage: float, desired coverage
            (0.1 for 10% of bp coverage; 1 for 100% bp coverage; 10 for 10x bp coverage).
    :param seed: int, random seed for reproducibility. Default is None.
    :return: n_frag x 2 matrix
    """

    # get fragment array
    fragments = _build_fragment_array(seq, sample_length, coverage, seed)

    # build matching taxid array
    taxids = _build_taxid_array(len(fragments), taxid)

    # combine arrays to build dataset
    combined = _combine_fragments_and_taxids(fragments, taxids)

    return combined


# tested
def _write_fragments(data, output_dir, i):
    """
    Writes data to given output directory as a binary numpy file.
    Filename format is 'fragments-xxxxx.npy' where x is the left-padded number i (i.e. fragments-00001.npy for i=1).

    :param output_dir: string, path where output should be written
    :param i: int, ith sequence currently being processed
    :param data: data to be written to file
    :return: None
    """
    # write fragment data to file
    output_file = '{}/fragments-{}.npy'.format(output_dir, str(i).zfill(5))
    with open(output_file, 'wb') as f:
        np.save(f, data)


# tested
def _create_fragment_directory(output_dir):
    """
    Creates fragment directory if it does not exist. Raises ValueError if it does exist to prevent overwriting of
    previous data.

    :param output_dir:
    :return: None
    """
    # check if output directory exists
    dir_exists = os.path.isdir(output_dir)

    if dir_exists:
        raise ValueError('Output directory already exists:', output_dir)
    else:
        os.mkdir(output_dir)


def generate_fragment_data(seq_file, taxid_file, output_dir, sample_length, coverage, seed=None):
    """
    Todo - process sequences in parallel.
    Todo - handle single taxid case
    Output directory cannot exist.

    :param seq_file:
    :param taxid_file:
    :param output_dir:
    :param sample_length:
    :param coverage:
    :param seed:
    :return:
    """
    # prepare output directory
    _create_fragment_directory(output_dir)

    # read taxid data
    taxids = np.loadtxt(taxid_file, dtype=str)

    # process each sequence
    for i, seq_record in enumerate(SeqIO.parse(seq_file, 'fasta')):
        results = _build_fragment_taxid_array(taxids[i], seq_record.seq, sample_length, coverage, seed)
        _write_fragments(results, output_dir, i)


def read_fragments(input_dir, pattern):
    """
    Reads all files in the input directory which follow given pattern and combines all data into a single array.
    Todo - ensure this works when starting with 1D arrays
    Todo - consider performance with large amount of fragment data

    :param input_dir: str, path to directory where fragments are stored
    :param pattern: str, unix-like pattern to match (i.e. '*.npy' for all files that end with .npy extension)
    :return: numpy matrix
    """
    # get list of fragment files
    fnames = glob(input_dir + '/' + pattern)

    # process list
    datasets = []
    for each in fnames:

        # read in file
        curr_frag = np.load(each)

        # determine whether to add contents to list
        if len(curr_frag) > 0:
            datasets.append(curr_frag)

    # combine arrays
    total = np.concatenate(datasets, axis=0)
    return total
