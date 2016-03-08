# ULDF
This is the code of ULDF. Paper of this code is: Yang Wang, Xinggang Wang, Wenyu Liu. Unsupervised Local Deep Feature for Image Recognition

Required libraries (included in the package):
1. VLFeat, by Andrea Vedaldi et al. 
2. LibLinear, by Machine Learning Group at National Taiwan University. 

How to make it work:
1. Download this code.
2. Download from http://yann.lecun.com/exdb/mnist the following 4 files:
          * train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz
          * t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz 
3. Unzip these 4 files by executing:
          * gunzip train-images-idx3-ubyte.gz
          * gunzip train-labels-idx1-ubyte.gz
          * gunzip t10k-images-idx3-ubyte.gz
          * gunzip t10k-labels-idx1-ubyte.gz 
      If unzipping with WinZip, make sure the file names have not been
      changed by Winzip. 
4. Run mnistdeepauto.m in matlab.
For any question, please contact: Yang Wang (wanyang@cslt.riit.tsinghua.edu.cn)
