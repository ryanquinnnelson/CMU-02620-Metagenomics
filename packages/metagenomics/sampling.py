"""
Defines sampling procedures for metagenomics data.
"""
from Bio import SeqIO
import numpy as np
import random


# tested
def split_record(seq_record):
    """
    Splits SeqRecord into id, sequence then calculates sequence length.
    :param seq_record: SeqRecord
    :return: (str, Seq, int) Tuple representing (sequence id, sequence, sequence length).
    """
    seq = seq_record.seq
    return seq_record.id, seq, len(seq)


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
def create_chararray(n_frag, sample_length):
    """
    Creates a numpy char array to contain fragments as they are generated, in addition to a flag indicating whether
    fragment is valid.

    :param n_frag: number of fragments to sample
    :param sample_length: length of each fragment
    :return: n x m matrix, where n is thenumber of fragments and m is sample_length + 1
    """
    charar = np.chararray((n_frag, sample_length + 1))
    charar[:] = '-'
    return charar


# tested
def get_random_position(seq_length, sample_length):
    """
    Selects a random start position for a sample, considering the length of the sequence and the length of the sample.
    Calculates the position after the sample end for slicing purposes.

    :param seq_length: length of sequence
    :param sample_length: length of fragment
    :return: (int, int) Tuple representing (start position, one position after end position).
    """
    start_pos = np.random.randint(0, seq_length - sample_length)
    one_after_end = start_pos + sample_length
    return start_pos, one_after_end


# tested
def convert_frag_seq(seq):
    """
    Converts sequence to lowercase numpy character array, adding a column to the end of the sequence for the
    validation flag.
    Todo - Determine if there is a faster way to do this.

    :param seq: Seq
    :return: m x 1 vector, character array of sequence
    """
    seq_lower = seq.lower
    split_seq = seq_lower.split()
    charar = np.array([split_seq, '-'])  # add column to end
    return charar


# tested
def fragment_is_valid(frag):
    allowed = ['a', 'c', 't', 'g']
    return all(c in allowed for c in frag)


# tested
def update_frag_array(frag, charar, is_valid, i):
    """
    Replaces ith row of character array with fragment and sets validation flag according to validity of fragment.
    If fragment is valid, sets last column of ith row to 'v'.
    If fragment is invalid, sets last column of ith row to 'i'.

    :param frag:
    :param charar:
    :param is_valid:
    :param i:
    :return:
    """
    # replace ith row with frag
    charar[i] = frag

    # set validation flag
    j = charar.shape[1] - 1  # last column
    if is_valid:
        charar[i][j] = 'v'
    else:
        charar[i][j] = 'i'

    return charar


# tested
def get_valid_rows(charar):
    """
    Determines which rows in the character array are invalid.
    Assumes invalid rows are marked by 'i' in their last column.

    :param charar: character array to check
    :return: array of invalid rows
    """
    n_cols = charar.shape[1]
    return np.where(charar[:, n_cols - 1] == b'v')


# tested
def remove_invalid_frags(charar):
    """
    Removes invalid rows from fragment array and removes validation flag column.

    :param charar: n x m matrix, where
    :return: (n - i) x (m - 1) matrix, where i is the number of invalid fragments
    """
    # get list of invalid rows
    valid = get_valid_rows(charar)
    charar = charar[valid]

    # remove validation flag column
    last_col = charar.shape[1] - 1
    charar = np.delete(charar, last_col, axis=1)

    return charar


# based on code from paper
def draw_fragments(seq_record, sample_length, coverage):
    """
    Assumes sequence is at least as long as than sample_length.
    Todo - Replace invalid samples with valid samples.

    :param seq_record: sequence to be sampled, expecting Bio.SeqRecord.SeqRecord
    :param sample_length: int, length of fragments to be drawn
    :param coverage: float, number of times each bp in the sequence is covered, on average.
    :param random_seed: int, seed for RNG. Provide a value for reproducibility.
    :return:
    """

    # compute number of fragments to draw
    seq_id, seq, seq_length = split_record(seq_record)
    n_frag = calc_number_fragments(seq_length, coverage, sample_length)

    frag_array = create_chararray(n_frag, sample_length)  # scaffold for fragments and validation flag
    for i in range(n_frag):
        # get fragment
        start_pos, one_after_end = get_random_position(seq_length, sample_length)
        frag_seq = seq[start_pos:one_after_end]

        # convert to lowercase character array
        frag = convert_frag_seq(frag_seq)

        # validate fragment doesn't contain unexpected characters
        is_valid = fragment_is_valid(frag)

        # update fragment array with chosen fragment
        frag_array = update_frag_array(frag, frag_array, is_valid, i)

    # remove invalid sequences from fragment array and remove validation column
    frag_array = remove_invalid_frags(frag_array)

    return frag_array

#
# def build_fragments(input_file, output_file, sample_length, coverage, random_seed=0):
#     """
#
#     :param input_file: File in which sequences are stored. .fasta format expected.
#     :param output_file:
#     :param sample_length:
#     :param coverage:
#     :param random_seed:
#     :return:
#     """
#
#     # initialize random seed
#     random.seed(random_seed)
#     np.random.seed(random_seed)
#
#     # for each sequence, draw fragments as needed
#     for i, seq_record in enumerate(SeqIO.parse(input_file, 'fasta')):
#         print('Building fragments for sequence {}...'.format(i))
#
#         # split record into id and sequence
#         genome_id = seq_record.id
#         seq = seq_record.seq
#         seq_length = len(seq)
#
#         # skip sequences which are shorter than sample_length
#         if seq_length >= sample_length:
#             # draw fragments
#             fragments = draw_fragments(lowercase_seq, sample_length, coverage, random_seed)
#
#             # save fragments to file
#             write_fragments_to_file(fragments, output_file)
#         else:
#             print('Sequence is skipped because it is not long enough.')
