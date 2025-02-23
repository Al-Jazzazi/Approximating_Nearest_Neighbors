import numpy as np
import sys
import os

# Converts a txt file into an fvecs file, assuming the txt file is in the format:
# label1 float11 float12 float13 ... float1n
# label2 float21 float22 float23 ... float2n
# ...
def txt_to_fvecs(txt_file):
    fvecs_file = os.path.split(os.path.abspath(txt_file))[0] + "/glove.fvecs"
    # Open txt and fvecs files
    with open(fvecs_file, 'wb') as fvecs:
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                # Turn each line in txt file into an array
                parts = line.strip().split()
                vector = np.array(parts[1:], dtype=float)
                # Write the array into fvecs
                fvecs.write(np.int32(len(vector)).tobytes())
                fvecs.write(vector.astype(np.float32).tobytes())

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 txt_to_fvecs.py <txt_filename>")
        return
    txt_to_fvecs(sys.argv[1])

if __name__ == '__main__':
    main()