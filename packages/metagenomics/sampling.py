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
    for sequence in SeqIO.parse(input_file, "fasta"):
        fragments = draw_fragments(sequence, L, coverage, random_seed=0)
        write_fragments_to_file(fragments, output_file)
