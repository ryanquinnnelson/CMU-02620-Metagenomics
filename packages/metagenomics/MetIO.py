def write_to_file(filename, output):

    with open(filename, "ba") as output_handle:
        np.savetxt(output_handle, output)