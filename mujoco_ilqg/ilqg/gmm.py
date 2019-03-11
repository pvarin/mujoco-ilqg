import numpy as np
import scipy as sp
import scipy.stats

def initialize_gmm(data, n_clusters):
    n_data = data.shape[0]
    data_mean = np.mean(data, axis=0)
    data_cov = np.einsum('ij,ik->jk', data, data)/n_data
    cluster_weights = np.ones(n_clusters)/n_clusters
    cluster_means = np.random.multivariate_normal(data_mean, data_cov, size=n_clusters)
    cluster_covs = np.tile(data_cov, (n_clusters, 1, 1))

    return cluster_weights, cluster_means, cluster_covs


def classify(data, weights, means, covs):
    p = cluster_pdfs(data, weights, means, covs)

    # Normalize the probilities.
    p /= np.sum(p, axis=0)
    return p

def maximize_likelihood(data, membership_weights):
    # Effective number of samples per cluster.
    n_eff = np.sum(membership_weights, axis=1)

    # Recompute the cluster weights.
    cluster_weights = n_eff/data.shape[0]
    means = np.sum(data * membership_weights[...,np.newaxis], axis=1)/n_eff[...,np.newaxis]
    centered_data = data - means[:,np.newaxis,:]
    covs = np.einsum('ij,ijk,ijl->ikl', membership_weights, centered_data, centered_data)/n_eff[...,np.newaxis,np.newaxis]
    return cluster_weights, means, covs

def cluster_pdfs(data, weights, means, covs):
    n_clusters = weights.shape[0]
    n_data = data.shape[0]
    p = np.empty((n_clusters, n_data))

    # Compute the probabilities for each of the clusters.
    for i in range(n_clusters):
        p[i,:] = weights[i]*sp.stats.multivariate_normal.pdf(data, means[i,:], covs[i,:,:], allow_singular=True)
    
    return p

def gmm_pdf(data, weights, means, covs):
    p = cluster_pdfs(data, weights, means, covs)
    p = np.sum(p, axis=0)
    return p

def fit_gmm(data, n_clusters=10, n_iters=100, n_restarts=1):
    '''
    Fits a Gaussian Mixture Model (GMM) to the data.
    '''

    # Track the best model so far.
    best_model = tuple()
    best_ll = -np.inf

    log_likelihood = np.empty((n_restarts, n_iters + 1))

    for i in range(n_restarts):
        # Initialize a random model.
        cluster_weights, cluster_means, cluster_covs = initialize_gmm(data, n_clusters)
        ll = gmm_pdf(data, cluster_weights, cluster_means, cluster_covs)
        ll = np.sum(np.log(ll))
        log_likelihood[i,0] = ll

        for j in range(n_iters):
            # Perform the expectation step.
            p = classify(data, cluster_weights, cluster_means, cluster_covs)
            
            # Perform the maximization step.
            cluster_weights, cluster_means, cluster_covs = maximize_likelihood(data, p)
            
            # Compute the log_likelihood.
            ll = gmm_pdf(data, cluster_weights, cluster_means, cluster_covs)
            ll = np.sum(np.log(ll))
            log_likelihood[i,j+1] = ll

            # Update the best model if necessary.
            if ll > best_ll:
                best_model = (cluster_weights, cluster_means, cluster_covs)
                best_ll = ll

    return best_model, log_likelihood