# Make exports directory
mkdir exports
cd exports
pip3 install h5py
# Download SIFT
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzvf sift.tar.gz
rm sift.tar.gz
# Download GIST
wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar -xzvf gist.tar.gz
rm gist.tar.gz
# Download GloVe
mkdir glove
cd glove 
wget http://ann-benchmarks.com/glove-25-angular.hdf5
python3 ../../conversions/convert_hdf5.py glove-25-angular.hdf5
rm glove-25-angular.hdf5
cd ..
# Download MNIST
mkdir mnist
cd mnist
wget http://ann-benchmarks.com/mnist-784-euclidean.hdf5
python3 ../../conversions/convert_hdf5.py mnist-784-euclidean.hdf5
rm mnist-784-euclidean.hdf5
cd ../..