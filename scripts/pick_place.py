from pathlib import Path
import time

import mujoco
import mujoco.viewer

import robot_learning_playground.envs.mujoco.utils as mjutils

ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "assets/pick_place/scene.xml"
#SCENE_PATH = ROOT / "assets/franka_emika_panda/scene.xml"

def main():
    # 1. Compile MJCF -> MjModel
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))

    # 2. Allocate dynamic memory for the state -> MjData
    data = mujoco.MjData(model)

    # 3. Inspect model
    mjutils.print_model_info(model)
    mjutils.print_joints(model)
    mjutils.print_bodies(model)
    mjutils.print_actuators(model)  
    mjutils.print_keyframes(model)

    # 4. Compute the initial state for keyframe 0 ("home")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # 5. Start the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            # Avanza la fisica di un timestep
            mujoco.mj_step(model, data)

            # Aggiorna il viewer
            viewer.sync()

            # Visualizzazione approssimativamente realtime
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()