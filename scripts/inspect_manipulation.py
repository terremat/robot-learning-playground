import time

import mujoco
import mujoco.viewer
import numpy as np


def get_grasp_contacts(model, data):
    cube_body_id = model.body("cube").id

    forces = {
        "left_finger": 0.0,
        "right_finger": 0.0,
    }

    counts = {
        "left_finger": 0,
        "right_finger": 0,
    }

    for i in range(data.ncon):
        contact = data.contact[i]

        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]

        if body1 == cube_body_id:
            other_body = body2
        elif body2 == cube_body_id:
            other_body = body1
        else:
            continue

        body_name = model.body(other_body).name

        if body_name not in forces:
            continue

        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)

        forces[body_name] += wrench[0]
        counts[body_name] += 1

    return forces, counts


model = mujoco.MjModel.from_xml_path(
    "assets/manipulation/scene.xml"
)
data = mujoco.MjData(model)

# Configurazione iniziale dal modello
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

q_home = data.qpos[:7].copy()

# Apriamo il gripper e lasciamolo stabilizzare
data.ctrl[:7] = data.qpos[:7].copy()
data.ctrl[7] = 255

finger1_id = model.joint("finger_joint1").id
finger2_id = model.joint("finger_joint2").id

finger1_adr = model.jnt_qposadr[finger1_id]
finger2_adr = model.jnt_qposadr[finger2_id]

for _ in range(500):
    mujoco.mj_step(model, data)

print("\n=== GRIPPER GEOMETRY ===")

for i in range(model.nbody):
    name = model.body(i).name

    if "finger" in name.lower() or "hand" in name.lower():
        print(name, data.xpos[i])


left_id = model.body("left_finger").id
right_id = model.body("right_finger").id

left_pos = data.xpos[left_id].copy()
right_pos = data.xpos[right_id].copy()

grasp_center = 0.5 * (left_pos + right_pos)

cube_joint_id = model.joint("cube_freejoint").id
cube_qpos_adr = model.jnt_qposadr[cube_joint_id]
cube_dof_adr = model.jnt_dofadr[cube_joint_id]

# Posizionamento del cubo
data.qpos[cube_qpos_adr:cube_qpos_adr + 3] = grasp_center
data.qpos[cube_qpos_adr + 3:cube_qpos_adr + 7] = [1, 0, 0, 0]
data.qvel[cube_dof_adr:cube_dof_adr + 6] = 0

# Test senza gravità
model.opt.gravity[:] = 0

mujoco.mj_forward(model, data)

print("Grasp center:", grasp_center)
print("Cube position:", data.xpos[model.body("cube").id])

data.ctrl[:7] = q_home
data.ctrl[7] = 0

mujoco.mj_step(model, data)

for i in range(data.ncon):
    contact = data.contact[i]

    geom1 = model.geom(contact.geom1).name
    geom2 = model.geom(contact.geom2).name

    if "cube" in geom1 or "cube" in geom2:
        print(
            f"{geom1} <-> {geom2}, "
            f"dist={contact.dist:.5f}"
        )

for i in range(data.ncon):
    contact = data.contact[i]

    for geom_id in [contact.geom1, contact.geom2]:
        geom = model.geom(geom_id)
        body_id = model.geom_bodyid[geom_id]

        print(
            f"geom_id={geom_id}, "
            f"geom_name={geom.name!r}, "
            f"body={model.body(body_id).name}, "
            f"group={model.geom_group[geom_id]}"
        )

    print("distance:", contact.dist)
    print("position:", contact.pos)
    print("---")

tcp_id = model.site("tcp").id

tcp_pos = data.site_xpos[tcp_id].copy()
tcp_rot = data.site_xmat[tcp_id].reshape(3, 3).copy()

print("TCP position:", tcp_pos)
print("TCP rotation:\n", tcp_rot)

print("TCP - grasp center:", tcp_pos - grasp_center)

tcp_id = model.site("tcp").id

tcp_pos = data.site_xpos[tcp_id].copy()
hand_pos = data.xpos[model.body("hand").id].copy()

direction = tcp_pos - hand_pos
direction /= np.linalg.norm(direction)

print("\n=== CUBE PLACEMENT TEST ===")

cube_geom_id = model.geom("cube_geom").id

for offset in [0.00, 0.01, 0.02, 0.03, 0.04]:
    candidate = grasp_center + offset * direction

    data.qpos[cube_qpos_adr:cube_qpos_adr + 3] = candidate
    data.qvel[cube_dof_adr:cube_dof_adr + 6] = 0

    mujoco.mj_forward(model, data)

    print(f"\n=== OFFSET {offset:.3f} ===")

    for i in range(data.ncon):
        contact = data.contact[i]

        if cube_geom_id not in (contact.geom1, contact.geom2):
            continue

        other_id = (
            contact.geom2
            if contact.geom1 == cube_geom_id
            else contact.geom1
        )

        body_id = model.geom_bodyid[other_id]

        print(
            f"body={model.body(body_id).name}, "
            f"geom={other_id}, "
            f"dist={contact.dist:.5f}"
        )


geom_id = 64

print("Type:", mujoco.mjtGeom(model.geom_type[geom_id]).name)
print("Size:", model.geom_size[geom_id])
print("Local position:", model.geom_pos[geom_id])
print("World position:", data.geom_xpos[geom_id])
print("Group:", model.geom_group[geom_id])
print("Contype:", model.geom_contype[geom_id])
print("Conaffinity:", model.geom_conaffinity[geom_id])

for body_name in ["left_finger", "right_finger"]:
    body_id = model.body(body_name).id

    print(f"\n=== {body_name} ===")

    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != body_id:
            continue

        print(
            f"geom={geom_id}, "
            f"type={mujoco.mjtGeom(model.geom_type[geom_id]).name}, "
            f"group={model.geom_group[geom_id]}, "
            f"pos={data.geom_xpos[geom_id]}, "
            f"size={model.geom_size[geom_id]}"
        )

left_pos = data.xpos[model.body("left_finger").id].copy()
right_pos = data.xpos[model.body("right_finger").id].copy()

opening_axis = left_pos - right_pos
opening_axis /= np.linalg.norm(opening_axis)

base_pos = grasp_center + 0.03 * direction

cube_geom_id = model.geom("cube_geom").id

print("\n=== LATERAL PLACEMENT TEST ===")

for lateral_offset in np.arange(-0.015, 0.016, 0.005):
    candidate = base_pos + lateral_offset * opening_axis

    data.qpos[cube_qpos_adr:cube_qpos_adr + 3] = candidate
    data.qvel[cube_dof_adr:cube_dof_adr + 6] = 0

    mujoco.mj_forward(model, data)

    contacts = []

    for i in range(data.ncon):
        contact = data.contact[i]

        if cube_geom_id not in (contact.geom1, contact.geom2):
            continue

        other_id = (
            contact.geom2
            if contact.geom1 == cube_geom_id
            else contact.geom1
        )

        body_name = model.body(
            model.geom_bodyid[other_id]
        ).name

        contacts.append((body_name, contact.dist))

    min_dist = min(
        (dist for _, dist in contacts),
        default=None,
    )

    bodies = sorted(set(name for name, _ in contacts))

    print(
        f"lateral={lateral_offset:+.3f} "
        f"contacts={len(contacts)} "
        f"bodies={bodies} "
        f"min_dist={min_dist}"
    )

cube_initial_pos = data.xpos[model.body("cube").id].copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < 5.0:

        data.ctrl[:7] = q_home
        data.ctrl[7] = 0  # Close gripper

        mujoco.mj_step(model, data)

        if data.time < 2.0:
            model.opt.gravity[:] = [0, 0, 0]
        else:
            model.opt.gravity[:] = [0, 0, -9.81]
        

        if int(data.time * 1000) % 100 == 0:
            finger1 = data.qpos[finger1_adr]
            finger2 = data.qpos[finger2_adr]

            cube_pos = data.xpos[model.body("cube").id]

            print(
                f"t={data.time:.2f} "
                f"opening={finger1 + finger2:.4f} "
                f"cube_disp={np.linalg.norm(cube_pos - cube_initial_pos):.4f} "
                f"contacts={data.ncon}"
            )

        if int(data.time * 1000) % 100 == 0:
            forces, counts = get_grasp_contacts(model, data)

            print(
                f"t={data.time:.2f} "
                f"opening={finger1 + finger2:.4f} "
                f"left={forces['left_finger']:.3f} N "
                f"right={forces['right_finger']:.3f} N "
                f"contacts={counts}"
            )

        viewer.sync()
        time.sleep(model.opt.timestep)