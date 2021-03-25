from packages.metagenomics import MetIO
import numpy as np


def test_build_taxid_array():
    taxid = 'NC_013451'
    n_frag = 2
    expected = np.array([b'NC_013451', b'NC_013451'])
    actual = MetIO._build_taxid_array(n_frag, taxid)
    np.testing.assert_array_equal(actual, expected)


def test_build_output_rows():
    fragments = np.array([b'atcg', b'gtcc'])
    taxid = 'NC_013451'
    expected = np.array([[b'NC_013451', b'atcg'],
                         [b'NC_013451', b'gtcc']])
    actual = MetIO._build_output_rows(fragments, taxid)
    np.testing.assert_array_equal(actual, expected)

# def test_split_record():
#     record = SeqRecord(
#         Seq("MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"),
#         id="YP_025292.1",
#         name="HokC",
#         description="toxic membrane protein, small")
#
#     expected_id = "YP_025292.1"
#     expected_seq = Seq("MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF")
#     expected_seq_length = 44
#
#     actual_id, actual_seq, actual_seq_length = sampling.split_record(record)
#     assert actual_id == expected_id
#     assert actual_seq == expected_seq
#     assert actual_seq_length == expected_seq_length
#
#
