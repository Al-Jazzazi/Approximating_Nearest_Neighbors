import sys
import os
import numpy as np

def read_fvecs(file):
    with open(file, 'rb') as f:
        data = []
        while True:
            vector_length = np.fromfile(f, dtype=np.int32, count=1)
            if len(vector_length) <= 0:
                print(vector_length)
                break
            vector = np.fromfile(f, dtype=np.float32, count=vector_length[0])
            data.append(vector)
        return data

def write_fvecs(file, vectors):
    with open(file, 'wb') as f:
        for vector in vectors:
            np.array([len(vector)], dtype=np.int32).tofile(f)
            vector.tofile(f)

def split_fvecs(input_file, output_file1, output_file2, output_size1, output_size2):
    vectors = read_fvecs(input_file)
    first = vectors[:(output_size1 + 1)]
    second = vectors[(output_size1 + 1):(output_size1 + output_size2 + 2)]
    write_fvecs(output_file1, first)
    write_fvecs(output_file2, second)

def main():
    if len(sys.argv) < 6:
        print("Usage: python3 split_fvecs.py <input_filename> <output_filename1> <output_filename2> <output_size1> <output_size2>")
        return
    split_fvecs(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))

if __name__ == '__main__':
    main()