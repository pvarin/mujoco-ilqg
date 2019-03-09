import os

import mujoco_py
import numpy as np

path = 'pendulum.xml'
path = os.path.join(os.path.dirname(__file__), "models", path)
model = mujoco_py.load_model_from_path(path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)
setattr(viewer.cam, 'distance', 4.0)

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


qpos = 0.06;
qvel = 0.0;
state = sim.get_state()
state = mujoco_py.MjSimState(state.time, qpos, qvel,
                             state.act, state.udd_state)
sim.set_state(state)
sim.forward()


for i in range(100):
    sim.step()
    viewer.render()
