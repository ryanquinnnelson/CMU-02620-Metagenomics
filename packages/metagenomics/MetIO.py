def write_to_file(filename, output):

    with open(filename, "ba") as output_handle:
        np.savetxt(output_handle, output)



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