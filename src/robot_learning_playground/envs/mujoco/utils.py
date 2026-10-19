import numpy as np
import mujoco

def print_model_info(model):
    print(f"Model:     {model.names}")
    print(f"Bodies:    {model.nbody}")
    print(f"Joints:    {model.njnt}")
    print(f"Actuators: {model.nu}")
    print(f"Geoms:     {model.ngeom}")
    print(f"Keyframes: {model.nkey}")
    print(f"Timestep:  {model.opt.timestep}")
    print(f"nq:        {model.nq}")
    print(f"nv:        {model.nv}")
    print(f"Integrator: {model.opt.integrator}")
    print(f"Solver:     {model.opt.solver}")
    print(f"Iterations: {model.opt.iterations}")
    print(f"Timestep:   {model.opt.timestep}")

def print_joints(model):
    print("\n=== JOINTS ===")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"{i:2d}  {name}")

def print_joint_addrs(model):
    print("\n=== JOINT ADDRESSES ===")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        qpos_addr = model.jnt_qposadr[i]
        qvel_addr = model.jnt_dofadr[i]
        print(f"{i:2d}  {name} | qpos_addr: {qpos_addr}, qvel_addr: {qvel_addr}")

def print_bodies(model):
    print("\n=== BODIES ===")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"{i:2d}  {name}")

def print_actuators(model):
    print("\n=== ACTUATORS ===")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        ctrl_range = model.actuator_ctrlrange[i]
        print(f"{i:2d}  {name} | ctrl_range: {ctrl_range[0]:.3f} to {ctrl_range[1]:.3f}")

def print_keyframes(model):
    print("\n=== KEYFRAMES ===")
    for i in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
        print(f"{i:2d}  {name}")

def print_sites(model):
    print("\n=== SITES ===")
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        print(f"{i:2d}  {name}")

def print_sensors(model):
    print("\n=== SENSORS ===")
    for i in range(model.nsensor):
        name = model.sensor(i).name
        print(f"{i:2d}  {name}")


################################
## Accessors
def get_qpos_ids(model, joint_names):
    return np.array([model.joint(joint_name).qposadr[0] for joint_name in joint_names])

def get_dof_ids(model, joint_names):
    return np.array([model.joint(joint_name).dofadr[0] for joint_name in joint_names])


################################
## Visualization
def draw_target(viewer, target):
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0

        mujoco.mjv_initGeom(
            scene.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.015, 0, 0],
            pos=target,
            mat=np.eye(3).flatten(),
            rgba=[0.2, 1.0, 0.2, 0.8],
        )
        scene.ngeom = 1