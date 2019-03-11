#!/usr/bin/env /usr/local/bin/python3

# Installed imports.
import numpy as np
import matplotlib.pyplot as plt

# Imports from this library.
from mujoco_ilqg.utils import (plot_gaussian, plot_gmm_pdf)
from mujoco_ilqg.ilqg.gmm import fit_gmm

# Local imports.
from test_datasets import (two_spirals, four_clusters)

def plot_results(data, model, log_likelihood):
    '''
    Plot the trained GMM and the log_likelihood during training.
    '''

    cluster_weights, cluster_means, cluster_covs = model

    # Clusters
    plt.figure()
    plt.scatter(data[:,0],data[:,1])
    for i in range(n_clusters):
        plot_gaussian(cluster_means[i,:], cluster_covs[i,:,:])

    # PDF
    plt.figure()
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    plot_gmm_pdf(cluster_weights, cluster_means, cluster_covs, range_min=data_min, range_max=data_max)

    # Log-likelihood
    plt.figure()
    plt.plot(log_likelihood.T)

if __name__ == '__main__':
    # Generate a dataset with 4 clusters.
    x, _ = four_clusters(1000, noise=0.1)

    # Fit the model.
    n_clusters = 4
    n_iters = 50
    n_restarts = 5
    model, log_likelihood = fit_gmm(x,
                                    n_clusters = n_clusters,
                                    n_iters = n_iters,
                                    n_restarts = n_restarts)
    
    # Plot the results.
    plot_results(x, model, log_likelihood)

    # Generate the spiral dataset.
    x, _ = two_spirals(1000, noise=1.0)

    # Fit the model.
    n_clusters = 20
    n_iters = 100
    n_restarts = 5
    model, log_likelihood = fit_gmm(x,
                                    n_clusters = n_clusters,
                                    n_iters = n_iters,
                                    n_restarts = n_restarts)

    # Plot the results.
    plot_results(x, model, log_likelihood)
    
    plt.show()
