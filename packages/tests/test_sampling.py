from Bio.Seq import Seq
from packages.metagenomics import sampling


def test_sequence_is_valid__success():
    my_seq = Seq("atcg")
    assert sampling.sequence_is_valid(my_seq)


def test_sequence_is_valid__failure():
    my_seq = Seq("atcgh")
    assert sampling.sequence_is_valid(my_seq) is False

