from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
import time

def KNN_without_GPU(X_train, y_train, X_test, y_test, k):
    start = time.clock()

    m = X_test.size(0)
    n = X_train.size(0)

    xx = (X_test ** 2).sum(dim=1, keepdim=True).expand(m, n)
    yy = (X_train ** 2).sum(dim=1, keepdim=True).expand(n, m).transpose(0, 1)

    dist_mat = xx + yy - 2 * X_test.matmul(X_train.transpose(0, 1))
    mink_idxs = dist_mat.argsort(dim=-1)
    res = []
    for idxs in mink_idxs:
        res.append(np.bincount(np.array([y_train[idx] for idx in idxs[:k]])).argmax())
    assert len(res) == len(y_test)
    # print("acc", accuracy_score(y_test, res))
    end = time.clock()
    time_elapsed = end - start
    print('the running time is : %s millisecond' % ((end - start)*1000))
  
if __name__ == "__main__":
  M = 1000
  N = 10
  test_gap = 800
  x = np.random.random((M, N))
  y = np.random.randint(0, 10, (M)).tolist()
  X_train = x[:test_gap]
  y_train = y[:test_gap]
  x_test = x[test_gap:]
  y_test = y[test_gap:]
  temp_X_train = []
  temp_X_test = []
  for i in range(len(X_train)):
      temp_X_train.append(torch.from_numpy(X_train[i]))
  for i in range(len(x_test)):
      temp_X_test.append(torch.from_numpy(x_test[i]))
  KNN_without_GPU(torch.stack(temp_X_train), y_train, torch.stack(temp_X_test), y_test, 5)
  
  