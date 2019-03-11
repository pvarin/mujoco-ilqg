import numpy as np

def two_spirals(n_points, noise=1.0):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * (2*np.pi)
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))


def four_clusters(n_points, noise=0.1):
    # Generate a dataset with 4 clusters.
    n_points = int(n_points/4)
    x1 = np.random.multivariate_normal([1, 1], noise*np.eye(2),size=n_points)
    x2 = np.random.multivariate_normal([-1, 1], noise*np.eye(2),size=n_points)
    x3 = np.random.multivariate_normal([1, -1], noise*np.eye(2),size=n_points)
    x4 = np.random.multivariate_normal([-1, -1], noise*np.eye(2),size=n_points)
    
    y1 = 0*np.ones_like(x1)
    y2 = 1*np.ones_like(x2)
    y3 = 2*np.ones_like(x3)
    y4 = 3*np.ones_like(x4)

    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])
    
    return x, y