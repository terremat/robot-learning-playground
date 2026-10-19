from dataclasses import dataclass
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


@dataclass
class Step:
    name: str
    position: np.ndarray
    gripper: int
    duration: float = 0.0

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

    gripper_id = model.actuator("panda_actuator8").id
    gripper_open = 0.04
    gripper_closed = 0.0

    #data.ctrl[gripper_id] = gripper_open


    # Print the initial state
    print(f"TCP:    {data.site('panda_tcp').xpos}")
    print(f"Object: {data.body('cube_red').xpos}")
    print(f"Target: {data.body('target_red').xpos}")

    cube_name = "cube_red"
    target_name = "target_red"
    pregrasp_height = 0.10

    # Compute pre-grasp and grasp pose
    cube_position = data.body(cube_name).xpos.copy()
    pregrasp_position = cube_position + np.array([0.0, 0.0, pregrasp_height])
    grasp_position = cube_position.copy()
    lift_position = grasp_position + np.array([0.0, 0.0, 0.10])

    target_position = data.body(target_name).xpos.copy()
    print(f"Target position: {target_position}")

    place_position = np.array([target_position[0], target_position[1], grasp_position[2]])
    target_above = place_position + np.array([0.0, 0.0, 0.10])
    
    R_initial = data.site_xmat[controller.tcp_id].reshape(3, 3).copy()
    target_rotation = R_initial.copy()

    print(f"Pre-grasp pose: {pregrasp_position}")
    print(f"Grasp pose: {grasp_position}")

    physics_dt = model.opt.timestep  # 0.002 s
    control_steps = 5
    control_dt = physics_dt * control_steps  # 0.01 s

    step_count = 0
    times = []
    tcp_positions = []
    tcp_errors = []
    joint_positions = []
    joint_commands = []

    close_elapsed = 0.0

    # Define the pick-and-place steps
    steps = [
        Step("PREGRASP",        pregrasp_position, 255),
        Step("DESCEND",         grasp_position,    255),
        Step("CLOSE",           grasp_position,    0, duration=0.0),
        Step("LIFT",            lift_position,     0),
        Step("MOVE_TO_TARGET",  target_above,      0),
        Step("PLACE",           place_position,    0),
        Step("OPEN",            place_position,    255, duration=0.0),
        Step("RETREAT",         target_above,      255),
    ]

    phase = "PREGRASP"

    step_index = 0
    step_elapsed = 0.0
    position_tolerance = 0.005
    rotation_tolerance = 0.05
    
    # Start the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:

        #mjutils.draw_target(viewer, pregrasp_position)
        #viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        #viewer.opt.sitegroup[0] = True

        while viewer.is_running():

            if step_index < len(steps):
                step = steps[step_index]

                # Apply gripper command
                data.ctrl[gripper_id] = step.gripper

                # Set target position and orientation for the controller                
                target_position = step.position
                error = controller.control_step(target_position, target_rotation, control_dt=control_dt)

                position_error = np.linalg.norm(error[:3])
                rotation_error = np.linalg.norm(error[3:])

                # Check completion
                if step.duration > 0:
                    completed = step_elapsed >= step.duration
                else:
                    completed = (
                        position_error < position_tolerance
                        and rotation_error < rotation_tolerance
                    )

                if completed:
                    print(f"{step.name} completed")

                    step_index += 1
                    step_elapsed = 0.0
                else:
                    step_elapsed += control_dt

            # Step the simulation
            for _ in range(control_steps):
                mujoco.mj_step(model, data)

            if step_count % 100 == 0:
                tcp_pos = data.site(0).xpos.copy()
                print(
                    f"t={data.time:.2f}s | "
                    f"TCP={np.round(tcp_pos, 3)} | "
                    f"error={np.linalg.norm(error[:3]):.4f}m"
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

    R_final = data.site_xmat[controller.tcp_id].reshape(3, 3).copy()

    print("Initial TCP Z:", R_initial[:, 2])
    print("Final TCP Z:", R_final[:, 2])

    R_error = R_initial.T @ R_final
    angle = np.arccos(
        np.clip((np.trace(R_error) - 1) / 2, -1.0, 1.0)
    )

    print("Orientation change:", np.degrees(angle), "deg")

    tcp_pos = data.site_xpos[controller.tcp_id]

    left_id = model.body("panda_left_finger").id
    right_id = model.body("panda_right_finger").id

    left_pos = data.xpos[left_id]
    right_pos = data.xpos[right_id]

    print("TCP:", tcp_pos)
    print("Left finger body origin:", left_pos)
    print("Right finger body origin:", right_pos)

    print("Left relative to TCP:", left_pos - tcp_pos)
    print("Right relative to TCP:", right_pos - tcp_pos)

    for body_name in ["panda_left_finger", "panda_right_finger"]:
        body_id = model.body(body_name).id

        print(f"\n{body_name}")

        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == body_id:
                print(
                    model.geom(geom_id).name,
                    "position:",
                    data.geom_xpos[geom_id],
                )

    # Check if the cube is at the desired position
    cube_position = data.body(cube_name).xpos.copy()
    desired_cube_position = np.array([
        target_position[0],
        target_position[1],
        0.122,
    ])

    placement_error = np.linalg.norm(
        cube_position - desired_cube_position
    )

    print("Final cube position:", cube_position)
    print("Placement error:", placement_error)

                
    times = np.array(times)
    tcp_positions = np.array(tcp_positions)
    joint_positions = np.array(joint_positions)
    joint_commands = np.array(joint_commands)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(times, tcp_positions[:, 0], label="TCP x")
    axes[0].plot(times, tcp_positions[:, 1], label="TCP y")
    axes[0].plot(times, tcp_positions[:, 2], label="TCP z")
    axes[0].axhline(pregrasp_position[0], linestyle="--", alpha=0.5)
    axes[0].axhline(pregrasp_position[1], linestyle="--", alpha=0.5)
    axes[0].axhline(pregrasp_position[2], linestyle="--", alpha=0.5)
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