"""
Defines sampling procedures for metagenomics data.
"""
from Bio import SeqIO
import numpy as np
import random


# tested
# based on code from paper
def calc_number_fragments(seq_length, coverage, sample_length):
    """
    Calculates the number of fragments to be randomly sampled from the given sequence in order to
    achieve required coverage.

    :param seq_length: length of the sequence
    :param coverage: float, percent of sequence coverage via sampling
    :param sample_length: length of each fragment
    :return: int, number of fragments
    """
    n_frag = seq_length * coverage / sample_length
    return round(n_frag)


# tested
def get_random_position(seq_length, sample_length):
    """
    Selects a random start position for a sample, considering the length of the sequence and the length of the sample.
    Calculates the position after the sample end for slicing purposes.

    :param seq_length: length of sequence
    :param sample_length: length of fragment
    :return: (int, int) Tuple representing (start position, one position after end position).
    """

    return np.random.randint(0, seq_length - sample_length)


# tested
def fragment_is_valid(frag):
    """
    Determines if fragment meets criteria required to be valid. Currently, the criteria is that all letters in the
    fragment are lowercase and are DNA nucleotides i.e. {a,c,t,g}.

    :param frag: Fragment selected from sequence
    :return: True if fragment is valid, false otherwise.
    """
    allowed = ['a', 'c', 't', 'g']
    return all(c in allowed for c in frag)


# tested
def draw_fragment(seq, sample_length):
    # choose random subsequence
    seq_length = len(seq)
    start_pos = get_random_position(seq_length, sample_length)

    # get fragment
    one_after_end = start_pos + sample_length
    frag_seq = seq[start_pos:one_after_end].lower()
    return str(frag_seq)


# tested
def build_fragment_array(seq, sample_length, n_frag):
    """

    :param seq:
    :param sample_length:
    :param n_frag:
    :return:
    """
    fragments = np.chararray((n_frag,), itemsize=sample_length)  # scaffold for fragments

    # draw fragments
    n_valid = 0
    n_failures = 0
    while n_valid < n_frag:

        # draw random fragment
        frag = draw_fragment(seq, sample_length)

        # determine whether to save or discard fragment
        if fragment_is_valid(frag):
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


# based on code from paper
# tested
def draw_fragments(seq, sample_length, coverage, seed):
    """
    Draws number of samples from sequence in order to achieve required coverage.

    :param seq: Sequence to be sampled
    :param sample_length: Length of sample
    :param coverage: Coverage value
    :return: n x 1 array, where n is the number of fragments drawn from sample
    """
    # initialize random seed
    np.random.seed(seed)

    # sample fragments if possible
    seq_length = len(seq)
    if seq_length >= sample_length:
        n_frag = calc_number_fragments(seq_length, coverage, sample_length)
        fragments = build_fragment_array(seq, sample_length, n_frag)
    else:
        fragments = None

    return fragments


# tested
def build_taxid_array(n_frag, taxid):
    taxid_length = len(taxid)
    taxids = np.chararray((n_frag,), itemsize=taxid_length)
    taxids[:] = taxid

    return taxids


# tested
def build_output_rows(fragments, taxid):
    n_frag = len(fragments)
    taxids = build_taxid_array(n_frag, taxid)
    return np.column_stack((taxids, fragments))

# # tested
# def split_record(seq_record):
#     """
#     Splits SeqRecord into id, sequence then calculates sequence length.
#
#     :param seq_record: SeqRecord
#     :return: (str, Seq, int) Tuple representing (sequence id, sequence, sequence length).
#     """
#     seq = seq_record.seq
#     return seq_record.id, seq, len(seq)


# def build_fragments(input_file, output_file, sample_length, coverage, random_seed=0):
#     """
#       Todo - consider parallel processing
#       Todo - consider writing separate files for each sequence in case file size is an issue
#     :param input_file: File in which sequences are stored. .fasta format expected.
#     :param output_file:
#     :param sample_length:
#     :param coverage:
#     :param random_seed:
#     :return:
#     """
#

#
#     # for each sequence, draw fragments as needed
#     for i, seq_record in enumerate(SeqIO.parse(input_file, 'fasta')):
#         print('Building fragments for sequence {}...'.format(i))
#
#         # skip sequences which are shorter than sample_length
#         seq_id, seq, seq_length = split_record(seq_record)
#         if seq_length >= sample_length:
#
#             # draw fragments
#             fragments = draw_fragments(seq, sample_length, coverage, random_seed)
#
#             # save fragments to file
#             write_fragments_to_file(fragments, output_file)
#         else:
#             print('Sequence is skipped because it is not long enough.')
