"""
Defines sampling procedures for metagenomics data.
"""
from Bio import SeqIO


def draw_fragments(sequence, L, coverage, random_seed=0):
    """


    :param L: int, length of fragments to be drawn
    :param coverage: float, number of times each bp in the sequence is covered, on average.
    :param random_seed: int, seed for RNG. Provide a value for reproducibility.
    :return:
    """
    pass


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

    # for each sequence, draw fragments as needed
    for i, seq_record in enumerate(SeqIO.parse(input_file, 'fasta')):
        print('Building fragments for sequence {}...'.format(i))
        lowercase_seq = seq_record.seq.lower
        if sequence_is_valid(lowercase_seq):
            fragments = draw_fragments(lowercase_seq, L, coverage, random_seed)
            write_fragments_to_file(fragments, output_file)
        else:
            print('Sequence {} is invalid.'.format(seq_record.id))
