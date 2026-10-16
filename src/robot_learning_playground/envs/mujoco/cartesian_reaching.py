import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = ROOT / "assets/franka_emika_panda/scene.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    tcp_id = model.site("tcp").id

    target = np.array([0.50, 0.10, 0.55])

    kp = 2.0
    max_cartesian_speed = 0.15  # m/s
    damping = 0.05
    max_joint_speed = 0.8       # rad/s

    control_dt = 0.01           # 100 Hz
    physics_dt = model.opt.timestep
    steps_per_control = round(control_dt / physics_dt)

    assert np.isclose(
        steps_per_control * physics_dt, control_dt
    )

    print("Physics timestep:", physics_dt)
    print("Control timestep:", control_dt)
    print("Target:  ", target)
    print("Current: ", data.site_xpos[tcp_id])

    q_ref = data.qpos[:7].copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            loop_start = time.perf_counter()

            # Update kinematics for the current joint state
            mujoco.mj_forward(model, data)

            # 1. Cartesian position error
            x = data.site_xpos[tcp_id].copy()
            error = target - x

            if np.linalg.norm(error) < 0.01:
                print("Target reached!")
                break

            # 2. Desired Cartesian velocity
            v_des = kp * error

            speed = np.linalg.norm(v_des)
            if speed > max_cartesian_speed:
                v_des *= max_cartesian_speed / speed

            # 3. Translational Jacobian
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))

            mujoco.mj_jacSite(
                model, data, jacp, jacr, tcp_id
            )

            J = jacp[:, :7]

            # 4. Damped least-squares inverse
            A = J @ J.T + damping**2 * np.eye(3)

            qdot_des = J.T @ np.linalg.solve(A, v_des)

            # Limit joint velocities
            qdot_des = np.clip(
                qdot_des,
                -max_joint_speed,
                max_joint_speed,
            )

            # 5. Convert velocity to position references
            q_des = data.qpos[:7].copy() + qdot_des * control_dt

            q_ref += qdot_des * control_dt

            # Respect actuator position limits
            q_des = np.clip(
                q_ref,
                model.actuator_ctrlrange[:7, 0],
                model.actuator_ctrlrange[:7, 1],
            )

            data.ctrl[:7] = q_des

            # 6. Advance physics for one control period
            for _ in range(steps_per_control):
                mujoco.mj_step(model, data)

            viewer.sync()

            joint_tracking_error = np.linalg.norm(
                q_ref - data.qpos[:7]
            )

            cartesian_error = np.linalg.norm(
                target - data.site_xpos[tcp_id]
            )

            print(
                f"\rCartesian error: {cartesian_error:.3f} m | "
                f"Joint tracking error: {joint_tracking_error:.3f} rad",
                end="",
                flush=True,
            )

            # Real-time visualization
            elapsed = time.perf_counter() - loop_start
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)


if __name__ == "__main__":
    main()