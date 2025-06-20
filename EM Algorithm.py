import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
X, y_true = make_blobs(n_samples=100, centers=4, cluster_std=0.60, random_state=0)
X = X[:, ::-1]  
gmm = GaussianMixture(n_components=4, random_state=42).fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)
print(probs[:5].round(3))
plt.figure(figsize=(8, 6))
size = 50 * probs.max(1) ** 2
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=size)
plt.title("GMM Clustering with Confidence Sizes")
plt.show()
def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = g
