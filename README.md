# ECE_GY_9413_KNN 

## code 
- `knn_cuda.cu` : Implementation of cuda-accelerated K nearest neighbors
- `test.cpp` : Test program of cuda knn
- `knn_sklearn.py` : Implementation of K nearest neighbors directly using sklearn
- `knn_pytorch_cpu.py` : K nearest neighbors in Pytorch(running on cpu)
- `knn_pytorch_gpu.py` : K nearest neighbors in Pytorch(running on gpu)

## Usage
Starting by `make`, and then `./test`  
You could change the size of reference numbers/query numbers/dimension in `test.cpp`, and redo the `make` and `./test` for running test program

## Note
The code on pytorch was deprecated since there is no K nearest neighbors library defined in pytorch directly. We don't think using hand-writting version comparing to CUDA and sklearn is a good choice, which may not represent the true running time of pytorch. 
