
import time

import numpy as np
import matplotlib

# Use the non-interactive "Agg" backend for matplotlib to avoid GUI issues in headless environments
matplotlib.use("Agg")

import mujoco
import mujoco.viewer

import matplotlib.pyplot as plt




model = mujoco.MjModel.from_xml_path(
    "assets/manipulation/scene.xml"
)
data = mujoco.MjData(model)

cube_id = model.body("cube").id
cube_weight = model.body_mass[cube_id] * abs(model.opt.gravity[2])

time_history = []
force_history = []
z_history = []

# Duration of the experiment in simulated seconds
duration = 2.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < duration:
        mujoco.mj_step(model, data)

        # Sum normal forces over all contacts
        normal_force = 0.0

        for i in range(data.ncon):
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, wrench)

            normal_force += wrench[0]

        # Save one sample per physics timestep
        time_history.append(data.time)
        force_history.append(normal_force)
        z_history.append(data.xpos[cube_id, 2])

        # Print every 0.1 seconds
        step = round(data.time / model.opt.timestep)

        if step % round(0.1 / model.opt.timestep) == 0:
            print(
                f"t={data.time:.2f} "
                f"z={data.xpos[cube_id, 2]:.5f} "
                f"contacts={data.ncon} "
                f"F_normal={normal_force:.4f} N "
                f"mg={cube_weight:.4f} N"
            )

        viewer.sync()
        time.sleep(model.opt.timestep)


# Plot after simulation
plt.figure(figsize=(10, 5))

plt.plot(
    time_history,
    force_history,
    label="Total normal contact force",
)

plt.axhline(
    y=cube_weight,
    color="red",
    linestyle="--",
    label="Cube weight (mg)",
)

plt.xlabel("Simulation time (s)")
plt.ylabel("Force (N)")
plt.title("Cube-table contact force")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/cube_contact_force.png")
plt.close()
