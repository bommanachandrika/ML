import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def kernel(point, xmat, k):
    m, n = np.shape(xmat)
    weights = np.mat(np.eye(m))  # Identity matrix for weights
    for j in range(m):
        diff = point - xmat[j]
        weights[j, j] = np.exp(diff @ diff.T / (-2.0 * k**2))
    return weights

def localWeight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    W = (xmat.T @ (wei @ xmat)).I @ (xmat.T @ (wei @ ymat.T))
    return W

def localWeightRegression(xmat, ymat, k):
    m, n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] @ localWeight(xmat[i], xmat, ymat, k)
    return ypred

# Load data
data = pd.read_csv('data10.csv')  # Make sure this CSV has columns: total_bill, tip
bill = np.array(data.total_bill)
tip = np.array(data.tip)

# Preparing data
mbill = np.mat(bill).T
mtip = np.mat(tip).T
m = mbill.shape[0]
one = np.mat(np.ones(m)).T
X = np.hstack((one, mbill))

# Set bandwidth k
k = 2
ypred = localWeightRegression(X, mtip, k)

# Sort X for plotting
SortIndex = X[:, 1].argsort(0)
xsort = X[SortIndex][:, 1]
ypred_sort = ypred[SortIndex.A1]  # .A1 flattens matrix to 1D

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(bill, tip, color='blue', label='Original Data')
plt.plot(xsort, ypred_sort, color='red', linewidth=2, label='LWLR Prediction')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.title('Locally Weighted Linear Regression')
plt.legend()
plt.show()





total_bill,tip
16.99,1.01
10.34,1.66
21.01,3.50
23.68,3.31
24.59,3.61
25.29,4.71
8.77,2.00
26.88,3.12
15.04,1.96
14.78,3.23
10.27,1.71
35.26,5.00
15.42,1.57
18.43,3.00
14.83,3.02
21.58,3.92
10.33,1.67
16.29,3.71
16.97,3.50
20.65,3.35

