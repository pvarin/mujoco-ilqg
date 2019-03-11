import os

import mujoco_py
import numpy as np

from mujoco_ilqg.utils import (
    get_state, set_state, reset_state
)

def quadratic_cost(Q, R, x_des):
    '''
    Generates quatdratic cost functions of the form:
        cost = .5(x-x_des)'Q(x-x_des) + u'Ru
    including the first and second order derivatives.
    '''
    def cost(x,u):
        x_err = x - x_des

        d2c_dx2 = Q
        dc_dx = Q.dot(x_err)
        d2c_du2 = R
        dc_du = R.dot(u)
        d2c_dxdu = np.zeros((len(x), len(u)))

        c = .5*x_err.T.dot(dc_dx) + .5*u.T.dot(dc_du)
        return c, dc_dx, dc_du, d2c_dx2, d2c_du2, d2c_dxdu
    return cost

def forward(sim, x_0, x_hat, u_hat, k, K, Q_uu_inv, viewer=None):
    N = k.shape[0]
    reset_state(sim, x_0)
    x_traj = np.empty_like(x_hat)
    u_traj = np.empty_like(u_hat)

    for i in range(N):
        # Stochastic feedback policy.
        x_traj[i,:] = get_state(sim)
        mean = u_hat[i,:] + k[i,:] + K[i,:,:].dot(x_traj[i,:]-x_hat[i,:])
        u_traj[i,:] = np.random.multivariate_normal(mean, Q_uu_inv[i,:,:])

        # Step the simulation.
        sim.data.ctrl[:] = u_traj[i,:]
        sim.step()

        if viewer is not None:
            viewer.render()

    return x_traj, u_traj

if __name__ == "__main__":
    # Setup the model, sim and viewer.
    path = 'pendulum.xml'
    path = os.path.join(os.path.dirname(__file__), "models", path)
    model = mujoco_py.load_model_from_path(path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    setattr(viewer.cam, 'distance', 4.0)

    # Define the cost function.
    nx = model.nq + model.nv
    nu = model.nu
    Q = np.eye(nx)
    R = np.eye(nu)
    x_des = np.array([np.pi, 0])
    cost = quadratic_cost(Q, R, x_des)

    # Run the simulation forward.
    n = 100
    x_init = np.array([0.0, 0.0])
    x_hat = np.zeros((n, nx))
    u_hat = np.zeros((n, nu))
    K = np.zeros((n, nu, nx))
    k = np.zeros((n, nu))
    Q_uu_inv = np.stack([np.eye(nu) for _ in range(n)])
    
    forward(sim, x_init, x_hat, u_hat, k, K, Q_uu_inv, viewer)
