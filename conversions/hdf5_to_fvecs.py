import sys
import os
import h5py
import numpy as np

# Converts HDF5 files to fvecs files
def hdf5_to_fvecs(hdf5_file):
    # Open the HDF5 file
    datasets = []
    with h5py.File(hdf5_file, "r") as f:
        # Get dataset names
        f.visititems(lambda name, obj: datasets.append(name) if isinstance(obj, h5py.Dataset) else None)
        for dataset in datasets:
          # Read the specified dataset
          data = f[dataset][:]
          data = data.astype(np.float32)
          # Write to fvecs file
          fvecs_file = os.path.split(os.path.abspath(hdf5_file))[0] + "/" + dataset + ".fvecs"
          with open(fvecs_file, "wb") as fvecs:
              for vector in data:
                  # Write the dimensionality (as int32)
                  dimension = np.int32(len(vector))
                  dimension.tofile(fvecs)
                  # Write the vector data
                  vector.tofile(fvecs)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 hdf5_to_fvecs.py <hdf5_filename>.\nFor better file organization, make sure the HDF5 file is in its own folder")
        return
    hdf5_to_fvecs(sys.argv[1])

if __name__ == '__main__':
    main()