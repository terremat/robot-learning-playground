import time

import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path(
    "assets/manipulation/scene.xml"
)
data = mujoco.MjData(model)

# Configurazione iniziale dal modello
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print("Bodies:", model.nbody)
print("Joints:", model.njnt)
print("Actuators:", model.nu)
print("Geometries:", model.ngeom)

print("\n=== ACTUATORS ===")

for i in range(model.nu):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i,
    )

    print(
        f"{i}: {name}, "
        f"ctrlrange={model.actuator_ctrlrange[i]}, "
        f"trntype={model.actuator_trntype[i]}"
    )

print("\n=== ACTUATOR DYNAMICS ===")

for i in range(model.nu):
    print(f"\n{model.actuator(i).name}")
    print("trntype:", model.actuator_trntype[i])
    print("gaintype:", model.actuator_gaintype[i])
    print("biastype:", model.actuator_biastype[i])
    print("gainprm:", model.actuator_gainprm[i, :3])
    print("biasprm:", model.actuator_biasprm[i, :3])
    print("trnid:", model.actuator_trnid[i])

print("\n=== TENDONS ===")

for i in range(model.ntendon):
    print("Name:", model.tendon(i).name)
    print("Range:", model.tendon_range[i])

print("\n=== JOINTS ===")

for i in range(model.njnt):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i,
    )

    print(
        f"{i}: {name}, "
        f"type={model.jnt_type[i]}, "
        f"qposadr={model.jnt_qposadr[i]}, "
        f"dofadr={model.jnt_dofadr[i]}"
    )

print("\n=== SITES ===")

for i in range(model.nsite):
    print(
        model.site(i).name,
        data.site_xpos[i],
    )

## Simulation

tcp_id = model.site("tcp").id

tcp_pos = data.site_xpos[tcp_id].copy()
tcp_rot = data.site_xmat[tcp_id].reshape(3, 3).copy()

print("TCP position:", tcp_pos)
print("TCP rotation:\n", tcp_rot)

# Manteniamo il braccio nella posizione iniziale
q_home = data.qpos[:7].copy()

finger1_id = model.joint("finger_joint1").id
finger2_id = model.joint("finger_joint2").id

finger1_adr = model.jnt_qposadr[finger1_id]
finger2_adr = model.jnt_qposadr[finger2_id]

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # Controller di posizione del braccio
        data.ctrl[:7] = q_home

        # Alterniamo i comandi del gripper ogni 2 secondi
        phase = int(data.time / 2) % 2
        data.ctrl[7] = 0 if phase == 0 else 255

        mujoco.mj_step(model, data)

        # Logging ogni 0.5 secondi
        if round(data.time, 3) % 0.5 < model.opt.timestep:
            print(
                f"t={data.time:.2f} "
                f"u={data.ctrl[7]:.0f} "
                f"finger1={data.qpos[finger1_adr]:.4f} "
                f"finger2={data.qpos[finger2_adr]:.4f}"
            )
            finger1 = data.qpos[finger1_adr]
            finger2 = data.qpos[finger2_adr]

            print(
                f"u={data.ctrl[7]:.0f}, "
                f"finger1={finger1:.4f}, "
                f"finger2={finger2:.4f}, "
                f"opening≈{finger1 + finger2:.4f} m"
            )

        viewer.sync()
        time.sleep(model.opt.timestep)

# Apriamo il gripper e lasciamolo stabilizzare
data.ctrl[:7] = data.qpos[:7].copy()
data.ctrl[7] = 255

for _ in range(500):
    mujoco.mj_step(model, data)

print("\n=== GRIPPER GEOMETRY ===")

for i in range(model.nbody):
    name = model.body(i).name

    if "finger" in name.lower() or "hand" in name.lower():
        print(name, data.xpos[i])