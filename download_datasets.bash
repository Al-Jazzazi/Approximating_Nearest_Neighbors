# Setup
mkdir exports
cd exports
pip3 install h5py

# Download SIFT (1000000 x 128)
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzvf sift.tar.gz
rm sift.tar.gz

# Download GIST (1000000 x 960)
wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar -xzvf gist.tar.gz
rm gist.tar.gz

# Download GloVe (1000000 x 100)
mkdir glove
cd glove 
wget http://ann-benchmarks.com/glove-100-angular.hdf5
python3 ../../utils/hdf5_to_fvecs.py glove-100-angular.hdf5
python3 ../../utils/split_fvecs.py train.fvecs glove_base.fvecs glove_learn.fvecs 1000000 100000
rm neighbors.fvecs distances.fvecs train.fvecs glove-100-angular.hdf5
mv test.fvecs glove_query.fvecs
cd ..

# Download Deep (1000000 x 256)
wget --no-check-certificate https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/deep1M.tar.gz
tar -xzvf deep1M.tar.gz
rm deep1M.tar.gz deep1M/*.lshbox
mv deep1M deep
cd deep
mv deep1M_base.fvecs deep_base.fvecs
mv deep1M_groundtruth.ivecs deep_groundtruth.ivecs
mv deep1M_learn.fvecs deep_learn.fvecs
mv deep1M_query.fvecs deep_query.fvecs
cd ../..