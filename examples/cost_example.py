import numpy as np
from mujoco_ilqg.utils import grad_check

Q = np.eye(2)
R = np.eye(1)
x_des = np.array([np.pi, 0])
def cost(x,u):
    x_err = x - x_des

    d2c_dx2 = Q
    dc_dx = Q.dot(x_err)
    d2c_du2 = R
    dc_du = R.dot(u)
    d2c_dxdu = np.zeros((len(x), len(u)))

    c = .5*x_err.T.dot(dc_dx) + .5*u.T.dot(dc_du)
    return c, dc_dx, dc_du, d2c_dx2, d2c_du2, d2c_dxdu


if __name__ == '__main__':
  # Validate the cost and derivatives
  x_0 = np.random.random(2)
  u_0 = np.random.random(1)
  grad_check(x_0,
             lambda x: cost(x, u_0)[0],
             lambda x: cost(x, u_0)[1])
  grad_check(x_0,
             lambda x: cost(x, u_0)[1],
             lambda x: cost(x, u_0)[3])
  grad_check(x_0,
             lambda x: cost(x, u_0)[0],
             lambda x: cost(x, u_0)[1])
  grad_check(u_0,
             lambda u: cost(x_0, u)[0],
             lambda u: cost(x_0, u)[2])
  grad_check(u_0,
             lambda u: cost(x_0, u)[2],
             lambda u: cost(x_0, u)[4])
  grad_check(u_0,
             lambda u: cost(x_0, u)[1],
             lambda u: cost(x_0, u)[5])