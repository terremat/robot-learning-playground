import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = ROOT / "assets/franka_emika_panda/scene.xml"


def main():

    # Load model
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # model -> describe physical properties of the robot
    # data  -> describe the current state of the robot (q, qdot, ctrl, etc.)

    # Initialize robot in home configuration
    home_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    # Reset robot to home configuration (set values also for actuators)
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data) #compute derived quantities (e.g., forward kinematics, etc.)

    print("Initial qpos:", data.qpos[:7])
    print("Initial ctrl:", data.ctrl[:7])

    # Joint target
    q_home = data.qpos[:7].copy()

    q_target = q_home.copy()
    q_target[0] += 0.5

    # Movement duration
    duration = 3.0

    # Simulation timestep
    dt = model.opt.timestep

    print(f"Physics timestep: {dt} s")

    # Open viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:

        # Compute a trajectory from home to target configuration in duration seconds
        # (This is a simple linear interpolation for demonstration purposes)

        while viewer.is_running():

            step_start = time.time()

            # Simulation time
            t = data.time

            # Interpolation factor in [0, 1] for the trajectory
            alpha = np.clip(t / duration, 0.0, 1.0)

            # Smooth interpolation
            alpha = 3 * alpha**2 - 2 * alpha**3

            # Desired joint positions
            q_desired = q_home + alpha * (q_target - q_home)

            # Send commands to position actuators
            data.ctrl[:7] = q_desired

            # Advance physics
            mujoco.mj_step(model, data)

            # Update viewer
            viewer.sync()

            # Print state every ~0.5 simulated seconds
            step_index = int(data.time / dt)

            if step_index % max(1, round(0.5 / dt)) == 0:

                print(
                    f"t={data.time:.2f} | "
                    f"q1_des={data.ctrl[0]:.3f} | "
                    f"q1={data.qpos[0]:.3f} | "
                    f"dq1={data.qvel[0]:.3f}"
                )

            # Synchronize with real time for visualization
            elapsed = time.time() - step_start

            print("q_ref:", data.ctrl[:7])
            print("q_actual:", data.qpos[:7])
            print("error:", data.ctrl[:7] - data.qpos[:7])
            print("qfrc_bias:", data.qfrc_bias[:7])
            print("qfrc_actuator:", data.qfrc_actuator[:7])

            if elapsed < dt:
                time.sleep(dt - elapsed)

            


if __name__ == "__main__":
    main()