from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = ROOT / "assets/franka_emika_panda/scene.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Initialize home configuration
    home_id = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    tcp_id = model.site("tcp").id

    # Translational and rotational Jacobians
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacSite(
        model,
        data,
        jacp,
        jacr,
        tcp_id,
    )

    # Only the seven arm joints
    J = jacp[:, :7]

    print("TCP position:", data.site_xpos[tcp_id])
    print("Jacobian shape:", J.shape)
    print("Jacobian:\n", np.round(J, 4))

    # Example joint velocity
    qdot = np.zeros(7)
    qdot[0] = 0.1  # rad/s

    # Cartesian velocity
    v_tcp = J @ qdot

    print("\nJoint velocity:", qdot)
    print("TCP velocity:", v_tcp, "m/s")

    # Small time interval
    dt = 0.001

    # Save current configuration and TCP position
    q_initial = data.qpos.copy()
    x_initial = data.site_xpos[tcp_id].copy()

    # Apply a small joint displacement
    data.qpos[:7] += qdot * dt

    # Recompute forward kinematics
    mujoco.mj_forward(model, data)

    x_new = data.site_xpos[tcp_id].copy()

    # Numerical Cartesian velocity
    v_numerical = (x_new - x_initial) / dt

    print("\nJacobian prediction:", v_tcp)
    print("Numerical velocity:  ", v_numerical)
    print("Error:", np.linalg.norm(v_tcp - v_numerical))

    # Restore original configuration
    data.qpos[:] = q_initial
    mujoco.mj_forward(model, data)


if __name__ == "__main__":
    main()