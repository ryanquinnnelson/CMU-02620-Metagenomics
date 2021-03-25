import csv

def write_fragments_to_file(filename, fragments, taxid, header=None):
    # open file handle
    with open(filename, "a") as output_handle:
        write = csv.writer(output_handle)

        if header:
            write.writerow(header)

        for row in fragments:
            output_row = taxid + row.tolist()
            write.writerow(output_row)