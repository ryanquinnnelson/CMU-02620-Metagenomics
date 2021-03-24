"""
Defines sampling procedures for metagenomics data.
"""
from Bio import SeqIO
import numpy as np
import random

# class Fragment():
#     def __init__(self, taxid, sequence):
#         self.taxid = taxid
#         self.sequence = sequence


# tested
# based on code from paper
def calc_number_fragments(seq_length, coverage, sample_length):

    n_frag = seq_length * coverage / sample_length
    return round(n_frag)


# based on code from paper
def draw_fragments(seq_record, sample_length, coverage):
    """

    :param seq_record: sequence to be sampled, expecting Bio.SeqRecord.SeqRecord
    :param sample_length: int, length of fragments to be drawn
    :param coverage: float, number of times each bp in the sequence is covered, on average.
    :param random_seed: int, seed for RNG. Provide a value for reproducibility.
    :return:
    """
    # split record into id and sequence
    genome_id = seq_record.id
    seq = seq_record.seq
    seq_length = len(seq)

    # compute number of fragments to draw
    n_frag = calc_number_fragments(seq_length, coverage, sample_length)

    # skip sequences which are shorter than sample_length
    if seq_length >= sample_length:
        fragments = None
    else:
        fragments = None

    return fragments





# # draw fragments if sequence is valid
# if sequence_is_valid(lowercase_seq):
#
# else:
#     print('Sequence {} is invalid.'.format(seq_record.id))


def write_fragments_to_file(fragments, filename):
    pass


# tested
def sequence_is_valid(sequence):
    """
    Checks whether sequence contains characters other than {a,c,t,g}.
    Todo - make this more efficient

    :param sequence: Expecting Bio.Seq.Seq with lowercase letters only.
    :return: True if sequence only contains expected nucleotides, false otherwise.
    """
    allowed = ['a', 'c', 't', 'g']
    return all(c in allowed for c in sequence)


def build_fragments(input_file, output_file, L, coverage, random_seed=0):
    """

    :param input_file: File in which sequences are stored. .fasta format expected.
    :param output_file:
    :param L:
    :param coverage:
    :param random_seed:
    :return:
    """

    # initialize random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    # for each sequence, draw fragments as needed
    for i, seq_record in enumerate(SeqIO.parse(input_file, 'fasta')):
        print('Building fragments for sequence {}...'.format(i))

        # convert to lowercase
        lowercase_seq = seq_record.seq.lower

        # draw fragments
        fragments = draw_fragments(lowercase_seq, L, coverage, random_seed)

        # save fragments to file
        write_fragments_to_file(fragments, output_file)
