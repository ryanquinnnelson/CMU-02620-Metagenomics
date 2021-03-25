"""
Defines sampling functionality for metagenomics data.
"""
import numpy as np
from Bio import SeqIO


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
    Chooses random subsequence of given sample_length within given sequence.

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
def _build_fragment_array(seq, sample_length, n_frag):
    """
    Draws required number of valid fragments from sequence and constructs array of results.
    Raises ValueError if too many invalid sequences are sampled in order to prevent an infinite loop in the case that
    the sequence does not contain valid subsequences of sample_length.

    :param seq: Bio.Seq.Seq, sequence to be sampled
    :param sample_length: int, length of samples
    :param n_frag: int, number of fragments to sample
    :return: n_frag x 1 array
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
            # update breakout counter
            n_failures += 1

        # determine if breakout is necessary
        if n_failures > n_frag * 10:
            raise ValueError(
                'Too many invalid fragments encountered. Sampling is stopped after {} attempts.'.format(n_failures))

    return fragments


# tested
def _draw_fragments_for_sequence(seq, sample_length, coverage):
    """
    Draws number of samples from sequence in order to achieve required coverage. Follows general sampling procedure
    laid out by Vervier et al. See https://arxiv.org/abs/1505.06915.

    :param seq: Bio.Seq.Seq, sequence to be sampled
    :param sample_length: int, length of samples
    :param coverage: float, desired coverage
            (0.1 for 10% of bp coverage; 1 for 100% bp coverage; 10 for 10x bp coverage).
    :param seed: random seed, for reproducibility
    :return: n x 1 array, where n is the number of fragments drawn from sample in order to meet required coverage.
            Returns empty array if sequence length is less than sample length.
    """

    # sample fragments if possible
    seq_length = len(seq)
    if seq_length >= sample_length:
        n_frag = _calc_number_fragments(seq_length, coverage, sample_length)
        fragments = _build_fragment_array(seq, sample_length, n_frag)
    else:
        fragments = np.empty(0,)

    return fragments


# tested
def _build_taxid_array(n_frag, taxid):
    taxid_length = len(taxid)
    taxids = np.chararray((n_frag,), itemsize=taxid_length)
    taxids[:] = taxid

    return taxids


# tested
def _build_fragment_rows_for_sequence(fragments, taxid):
    n_frag = len(fragments)
    taxids = _build_taxid_array(n_frag, taxid)
    rows = np.column_stack((taxids, fragments))
    return rows


# Todo - process sequences in parallel.
def draw_fragments(seq_file, taxid_file, output_dir, sample_length, coverage, seed=None):

    if seed:
        np.random.seed(seed)  # initialize random seed

    # read in taxids
    taxids = np.loadtxt(taxid_file)

    # read in sequences
    for i, seq_record in enumerate(SeqIO.parse(seq_file, 'fasta')):
        # build fragment data
        fragments = _draw_fragments_for_sequence(seq_record.seq, sample_length, coverage)

        if len(fragments) > 0:
            rows = _build_fragment_rows_for_sequence(fragments, taxids[i])

            # write fragment data to file
            output_file = '{}/fragments-{}.npy'.format(output_dir, str(i).zfill(5))
            with open(output_file, 'wb') as f:
                np.save(f, rows)
