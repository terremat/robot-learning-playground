from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import robot_learning_playground.envs.mujoco.utils as mjutils
from robot_learning_playground.envs.mujoco.cartesian_controller import CartesianController

ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "assets/pick_place/scene.xml"
#SCENE_PATH = ROOT / "assets/franka_emika_panda/scene.xml"

def get_pregrasp_position(data, cube_name, height=0.12):
    cube_pos = data.body(cube_name).xpos.copy()
    return cube_pos + np.array([0.0, 0.0, height])

def main():
    # Compile MJCF -> MjModel
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))

    # Allocate dynamic memory for the state -> MjData
    data = mujoco.MjData(model)

    # Inspect model
    mjutils.print_model_info(model)
    mjutils.print_joints(model)
    mjutils.print_bodies(model)
    mjutils.print_actuators(model)  
    mjutils.print_keyframes(model)
    mjutils.print_sites(model)
    mjutils.print_joint_addrs(model)

    # Compute the initial state for keyframe 0 ("home")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    controller = CartesianController(
        model,
        data,
        prefix="panda",
        tcp_name="tcp",
    )
    controller.reset()

    # Print the initial state
    print(f"TCP:    {data.site('panda_tcp').xpos}")
    print(f"Object: {data.body('cube_red').xpos}")
    print(f"Target: {data.body('target_red').xpos}")

    # Compute pre-grasp pose
    pre_grasp_pos = get_pregrasp_position(data, 'cube_red', height=0.1)
    print(f"Pre-grasp pose: {pre_grasp_pos}")
    
    physics_dt = model.opt.timestep  # 0.002 s
    control_steps = 5
    control_dt = physics_dt * control_steps  # 0.01 s

    step_count = 0
    times = []
    tcp_positions = []
    tcp_errors = []
    joint_positions = []
    joint_commands = []

    # Start the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:

        mjutils.draw_target(viewer, pre_grasp_pos)

        while viewer.is_running():

            error = controller.control_step(pre_grasp_pos, control_dt=control_dt)

            for _ in range(control_steps):
                mujoco.mj_step(model, data)

            if step_count % 100 == 0:
                tcp_pos = data.site(0).xpos.copy()
                print(
                    f"t={data.time:.2f}s | "
                    f"TCP={np.round(tcp_pos, 3)} | "
                    f"error={error:.4f}m"
                )

            # Log data
            times.append(data.time)
            tcp_positions.append(data.site(0).xpos.copy())
            tcp_errors.append(error)
            joint_positions.append(data.qpos[controller.qpos_ids].copy())
            joint_commands.append(controller.qpos_cmd.copy())

            # Update viewer
            viewer.sync()
            time.sleep(control_dt)
            step_count += 1
    
    times = np.array(times)
    tcp_positions = np.array(tcp_positions)
    joint_positions = np.array(joint_positions)
    joint_commands = np.array(joint_commands)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(times, tcp_positions[:, 0], label="TCP x")
    axes[0].plot(times, tcp_positions[:, 1], label="TCP y")
    axes[0].plot(times, tcp_positions[:, 2], label="TCP z")
    axes[0].axhline(pre_grasp_pos[0], linestyle="--", alpha=0.5)
    axes[0].axhline(pre_grasp_pos[1], linestyle="--", alpha=0.5)
    axes[0].axhline(pre_grasp_pos[2], linestyle="--", alpha=0.5)
    axes[0].set_ylabel("TCP position [m]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(times, joint_positions[:, 0], label="Joint 1 actual")
    axes[1].plot(times, joint_commands[:, 0], "--", label="Joint 1 command")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Joint position [rad]")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig("outputs/pick_place_debug.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()