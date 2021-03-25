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
    Creates a numpy char array to contain fragments as they are generated.

    :param n_frag: number of fragments to sample
    :param sample_length: length of each fragment
    :return: n x L matrix, where n is the number of fragments and L is sample_length
    """
    charar = np.chararray((n_frag, sample_length))
    charar[:] = '!'
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
def convert_frag_seq(frag_seq):
    """
    Converts sequence to lowercase numpy character array, adding a column to the end of the sequence for the
    validation flag.
    Todo - Determine if there is a faster way to do this.

    :param frag_seq: Seq
    :return: m x 1 vector, character array of sequence
    """

    seq_lower = frag_seq.lower()
    split_seq = list(seq_lower)
    charar = np.array(split_seq, dtype='|S1')
    return charar


# tested
def fragment_is_valid(frag):
    print(frag)
    allowed = [b'a', b'c', b't', b'g']
    return all(c in allowed for c in frag)


# tested
# based on code from paper
def draw_fragments(seq_record, sample_length, coverage):
    """
    Assumes sequence is at least as long as than sample_length.
    Todo - Consider using numeric array to represent fragments instead of chars. (more space efficient)

    :param seq_record: sequence to be sampled, expecting Bio.SeqRecord.SeqRecord
    :param sample_length: int, length of fragments to be drawn
    :param coverage: float, number of times each bp in the sequence is covered, on average.
    :param random_seed: int, seed for RNG. Provide a value for reproducibility.
    :return:
    """

    # compute number of fragments to draw
    seq_id, seq, seq_length = split_record(seq_record)
    n_frag = calc_number_fragments(seq_length, coverage, sample_length)
    frag_array = create_chararray(n_frag, sample_length)  # scaffold for fragments

    # draw fragments
    n_valid = 0
    while n_valid < n_frag:
        # get fragment
        start_pos, one_after_end = get_random_position(seq_length, sample_length)
        frag_seq = seq[start_pos:one_after_end]

        # convert to lowercase character array
        frag = convert_frag_seq(frag_seq)

        # determine whether to save or discard fragment
        if fragment_is_valid(frag):
            frag_array[n_valid] = frag
            n_valid += 1
            print(frag_array)

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
