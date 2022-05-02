from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time

nb = 59000
dim = 500
ref_nb= 50000
X = np.random.random((nb, dim))
y = np.random.randint(low=1, high=10, size=(nb,1))
X_train = X[:ref_nb]
X_test = X[ref_nb:]
y_train = y[:ref_nb]
y_test = y[ref_nb:]

neigh = KNeighborsClassifier(n_neighbors=5)
elapsed = 0
for i in range(5):
    neigh.fit(X_train, y_train)
    T1 = time.perf_counter()
    y_pred = neigh.predict(X_test)
    T2 = time.perf_counter()
    elapsed += T2 - T1
print('sklearn predict time comsuming : %s s' % (elapsed / 5))