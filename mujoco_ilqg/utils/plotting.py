import numpy as np
import matplotlib.pyplot as plt

from mujoco_ilqg.ilqg.gmm import gmm_pdf

def plot_gmm_pdf(cluster_weights, cluster_means, cluster_covs, range_min=None, range_max=None):
    if range_min is None:
        range_min = np.min(cluster_means, axis=0)
    if range_max is None:
        range_max = np.max(cluster_means, axis=0)

    x_range = np.linspace(range_min[0], range_max[0], 100)
    y_range = np.linspace(range_min[1], range_max[1], 100)
    X, Y = np.meshgrid(x_range, y_range)
    x = np.stack([X.flatten(), Y.flatten()], axis=-1)
    p = gmm_pdf(x, cluster_weights, cluster_means, cluster_covs)
    plt.pcolor(X, Y, p.reshape(X.shape))

def plot_gaussian(mean, cov):
    # Generate a circle at the origin with unit radius.
    t = np.linspace(0,2*np.pi)
    x = np.vstack([np.sin(t), np.cos(t)]).T

    # Transform and shift according to the mean and covariance.
    L = np.linalg.cholesky(cov)
    x = x.dot(L.T) + mean
    
    # Plot the distribution.
    plt.plot(x[:,0], x[:,1])