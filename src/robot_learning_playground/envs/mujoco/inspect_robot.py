from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[4]

MODEL_PATH = (
    ROOT
    / "assets"
    / "franka_emika_panda"
    / "scene.xml"
)


def main():

    # 1. Load model
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    # 2. Create simulation state
    data = mujoco.MjData(model)

    # 3. Initialize derived quantities
    mujoco.mj_forward(model, data)

    print("=== MuJoCo Model ===")

    print(f"Bodies:    {model.nbody}")
    print(f"Joints:    {model.njnt}")
    print(f"Actuators: {model.nu}")
    print(f"nq:        {model.nq}")
    print(f"nv:        {model.nv}")

    print("\n=== Initial State ===")

    print("qpos:", data.qpos)
    print("qvel:", data.qvel)
    print("ctrl:", data.ctrl)

    print("\n=== Joints ===")

    for i in range(model.njnt):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            i,
        )

        print(i, name)

    print("\n=== Actuators ===")

    for i in range(model.nu):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            i,
        )

        print(i, name)

    print("\n=== Actuator Details ===")

    for i in range(model.nu):

        actuator_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            i,
        )

        transmission_type = model.actuator_trntype[i]
        transmission_id = model.actuator_trnid[i]

        print(f"\nActuator {i}: {actuator_name}")
        print(f"  Transmission type: {transmission_type}")
        print(f"  Transmission ID:   {transmission_id}")
        print(f"  Ctrl range:        {model.actuator_ctrlrange[i]}")
        print(f"  Gain parameters:   {model.actuator_gainprm[i][:3]}")
        print(f"  Bias parameters:   {model.actuator_biasprm[i][:3]}")

    print("\n=== Keyframes ===")

    for i in range(model.nkey):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_KEY,
            i,
        )

        print(f"Keyframe {i}: {name}")
        print("qpos:", model.key_qpos[i])
        print("ctrl:", model.key_ctrl[i])


if __name__ == "__main__":
    main()