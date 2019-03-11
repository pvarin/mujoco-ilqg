import numpy as np
import mujoco_py

def set_state(sim, x):
    state = sim.get_state()
    qpos = x[:sim.model.nq]
    qvel = x[sim.model.nq:]
    state = mujoco_py.MjSimState(state.time, qpos, qvel,
                                 state.act, state.udd_state)
    sim.set_state(state)
    sim.forward()

def reset_state(sim, x_0):
    sim.reset()
    set_state(sim, x_0)

def get_state(sim):
    return np.concatenate([sim.data.qpos, sim.data.qvel])