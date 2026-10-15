from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = ROOT / "assets/franka_emika_panda/scene.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Initialize home configuration
    home_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    mujoco.mj_resetDataKeyframe(model, data, home_id)

    # Compute forward kinematics
    mujoco.mj_forward(model, data)

    print("=== Bodies ===")

    for i in range(model.nbody):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            i,
        )

        position = data.xpos[i]

        print(f"{i:2d} | {name:20s} | xyz = {position}")

    print("\n=== Sites ===")

    for i in range(model.nsite):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            i,
        )

        position = data.site_xpos[i]

        print(f"{i:2d} | {name:20s} | xyz = {position}")


if __name__ == "__main__":
    main()