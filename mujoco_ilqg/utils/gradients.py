import numpy as np

def grad_check(x_0, fcn, grad_fcn, eps=1e-6, atol=1e-6):
    f = fcn(x_0)
    f_x = grad_fcn(x_0)

    assert f.shape == f_x.shape[:-1], 'the dimension of the function must match the leading dimensions of the gradient'
    assert x_0.size == f_x.shape[-1], 'the dimension of x must match the last dimension of the gradient'
    
    f_x_hat = np.zeros_like(f_x)
    for i in range(x_0.size):
        dx = np.zeros_like(x_0)
        dx[i] = eps
        df = fcn(x_0 + dx) - f
        f_x_hat[...,i] = df/eps

    assert np.allclose(f_x, f_x_hat, atol=1e-6), 'the gradients do not match to the specified tolerance'
