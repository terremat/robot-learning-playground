import time
import numpy as np

import mujoco
import mujoco.viewer

import robosuite as suite
from robosuite import load_controller_config


controller_config = load_controller_config(
    default_controller="OSC_POSE"
)

"""
Robosuite uses a different rendering system. 
Lets use Mujoco's built-in viewer to visualize the environment.
"""
env = suite.make(
    env_name="Lift",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
    control_freq=20,
    horizon=200,
)

obs = env.reset()

# 1. Observation space
print("\n=== OBSERVATIONS ===")
for name, value in obs.items():
    print(f"{name:35s} shape={np.shape(value)}")

# 2. Action space
print("\n=== ACTION SPACE ===")
print("Action dimension:", env.action_dim)
print("Action limits:", env.action_spec)

# 3. MuJoCo model
model = env.sim.model

print("\n=== MUJOCO MODEL ===")
print("Bodies:", model.nbody)
print("Joints:", model.njnt)
print("Actuators:", model.nu)
print("Sensors:", model.nsensor)
print("Cameras:", model.ncam)
print("Geometries:", model.ngeom)

# 4. Bodies
print("\n=== BODIES ===")
for name in model.body_names:
    print(name)

# 5. Actuators
print("\n=== ACTUATORS ===")
for name in model.actuator_names:
    print(name)

# 6. Simulation

# robosuite 1.4.0 wraps the native MuJoCo model and data
model = env.sim.model._model
data = env.sim.data._data

print("\n=== SIMULATION ===")
for body_name in ["table", "cube_main"]:
    body_id = model.body(body_name).id

    print(f"\n=== {body_name} ===")
    print("Position (world):", data.xpos[body_id])
    print("Mass:", model.body_mass[body_id])

    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != body_id:
            continue

        print("\nGeom:", mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        ))
        print("Type:", model.geom_type[geom_id])
        print("Size:", model.geom_size[geom_id])
        print("Friction:", model.geom_friction[geom_id])
        print("Collision group:", model.geom_group[geom_id])

    for joint_id in range(model.njnt):
        if model.jnt_bodyid[joint_id] == body_id:
            print("\nJoint:", mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            ))
            print("Joint type:", model.jnt_type[joint_id])


"""
print("\n=== VIEWER ===")
with mujoco.viewer.launch_passive(model, data) as viewer:

    #disable collision geometry visualization (geomgroup 0)
    # - visual geometry (geomgroup 1), and contact visualization (geomgroup 2)
    viewer.opt.geomgroup[0] = 0
    
    for step in range(200):
        action = np.zeros(env.action_dim)

        obs, reward, done, info = env.step(action)

        viewer.sync()
        time.sleep(1.0 / env.control_freq)

        if done or not viewer.is_running():
            break

print("\n=== SIMULATION STEP ===")
print("Reward:", reward)
print("Done:", done)
print("Simulation time:", env.sim.data.time)
"""

env.close()
