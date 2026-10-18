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

def print_joints(model):
    print("\n=== JOINTS ===")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"{i:2d}  {name}")

def print_bodies(model):
    print("\n=== BODIES ===")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"{i:2d}  {name}")

def print_actuators(model):
    print("\n=== ACTUATORS ===")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"{i:2d}  {name}")

def print_keyframes(model):
    print("\n=== KEYFRAMES ===")
    for i in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
        print(f"{i:2d}  {name}")