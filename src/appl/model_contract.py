"""Fixed MJCF structural checks reused from the audited reconstruction contract."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .journal import safe
from .io import digest as file_sha

def parse_xml(text):
    if "<!" in text or "<?" in text:
        raise ValueError(
            "XML declarations, entities and external resources are forbidden"
        )
    root = ET.fromstring(text)
    if root.tag != "mujoco":
        raise ValueError("Expected MJCF mujoco root")
    if any(
        root.find(".//" + tag) is not None
        for tag in (
            "include",
            "extension",
            "plugin",
            "composite",
            "flexcomp",
            "replicate",
        )
    ):
        raise ValueError("Only explicit flat MJCF; no includes/plugins/generators")
    return root


def edited_xml(content,edits):
    """Validate the complete edit batch in memory before any file mutation."""
    model=parse_xml(content)
    if not isinstance(edits,list) or not 1<=len(edits)<=64:raise ValueError('Use 1–64 explicit edits')
    for edit in edits:
        nodes=model.findall(edit['selector']) if edit['selector']!='.' else [model]
        if not nodes:raise ValueError('Selector did not match: '+edit['selector'])
        for node in nodes:
            if edit['operation']=='set':node.attrib.update(edit['attributes'])
            elif edit['operation']=='append':node.append(ET.fromstring(edit['xml']))
            elif edit['operation']=='remove':
                parents={child:parent for parent in model.iter() for child in parent};parents[node].remove(node)
            else:raise ValueError('Unknown XML operation')
    text=ET.tostring(model,encoding='unicode');parse_xml(text)
    return text


def materialize(source, assets, manifest, destination):
    root = parse_xml(Path(source).read_text())
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("usethread", "false")
    for key in ("meshdir", "texturedir", "assetdir"):
        compiler.attrib.pop(key, None)
    for item in root.iter():
        if "file" in item.attrib:
            name = item.get("file")
            if not name.startswith("asset://"):
                raise ValueError("Model files must use an explicit asset:// capability")
            name = name[len("asset://") :]
            if name not in manifest:
                raise ValueError("Unknown asset capability")
            path = safe(assets, name)
            if file_sha(path) != manifest[name]:
                raise ValueError("Asset changed")
            item.set("file", str(path.resolve()))
    ET.ElementTree(root).write(destination, encoding="unicode")


def audit_model(m, public):
    native = public["native"]

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def jid(name):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)

    def bid(name):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)

    def gid(name):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)

    require(
        (m.nq, m.nv, m.nu) == (24, 22, 9), "Native mapping requires nq=24,nv=22,nu=9"
    )
    require(m.nmocap == 0, "No mocap bodies")
    require(np.allclose(m.opt.gravity, [0, 0, -9.81]), "Fixed public gravity")
    require(abs(m.opt.timestep - 0.002) < 1e-10, "Fixed integration period .002 s")
    expected = [f"joint{i}" for i in range(1, 8)] + [
        "finger_joint1",
        "finger_joint2",
        "drawer_slide",
        "red_joint",
        "blue_joint",
    ]
    require(
        [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        == expected,
        "Fixed joint ordering/names",
    )
    require(
        list(m.jnt_type[-3:])
        == [
            mujoco.mjtJoint.mjJNT_SLIDE,
            mujoco.mjtJoint.mjJNT_FREE,
            mujoco.mjtJoint.mjJNT_FREE,
        ],
        "Drawer slide plus two free objects required",
    )
    require(
        np.allclose(m.jnt_axis[jid("drawer_slide")], [-1, 0, 0]),
        "Drawer travel must be -x",
    )
    require(
        np.allclose(m.jnt_range[jid("drawer_slide")], [0, 0.30]),
        "Public drawer encoder travel 0..0.30 m",
    )
    require(
        np.allclose(m.body_pos[bid("link0")], public["robot_base"]),
        "Fixed measured robot mounting pose",
    )
    require(
        np.allclose(m.actuator_trnid[:, 0], range(9)),
        "Only seven arm and two finger actuators, in order",
    )
    require(
        np.allclose(m.actuator_gainprm[:, 0], 1000)
        and np.allclose(m.actuator_biasprm[:, 1], -1000)
        and np.allclose(m.actuator_biasprm[:, 2], -100),
        "Native kp=1000,kd=100 required",
    )
    require(
        np.allclose(m.actuator_forcerange, np.tile([-100, 100], (9, 1))),
        "Native actuator force bounds",
    )
    require(
        np.allclose(m.jnt_range[:7, 0], native["action"]["low"][:7], atol=1e-5)
        and np.allclose(m.jnt_range[:7, 1], native["action"]["high"][:7], atol=1e-5),
        "Native arm position ranges",
    )
    require(
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp") >= 0,
        "Named tcp site required",
    )
    require(
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front") >= 0,
        "Named front camera required",
    )

    def box(name, body, position, half):
        g = gid(name)
        require(
            g >= 0 and m.geom_bodyid[g] == body,
            "Missing/misbound fixed CAD box " + name,
        )
        require(
            m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
            and np.allclose(m.geom_size[g], half, atol=1e-8)
            and np.allclose(m.geom_pos[g], position, atol=1e-8),
            "Known object asset dimensions changed: " + name,
        )
        require(
            m.geom_contype[g] > 0 and m.geom_conaffinity[g] > 0,
            "Disabled collision " + name,
        )

    table = gid("table")
    require(
        table >= 0 and m.geom_bodyid[table] == 0 and m.geom_contype[table] > 0,
        "Reconstruct a physical static table",
    )
    drawer_geoms = [
        i
        for i in range(m.ngeom)
        if m.geom_bodyid[i] == bid("drawer") and m.geom_contype[i] > 0
    ]
    cabinet_geoms = [
        i
        for i in range(m.ngeom)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
            "cabinet_"
        )
        and m.geom_contype[i] > 0
    ]
    require(
        len(drawer_geoms) >= 6 and len(cabinet_geoms) >= 3,
        "Drawer floor/walls/handle and static cabinet must be reconstructed",
    )
    require(
        0.2 <= m.body_mass[bid("drawer")] <= 20,
        "Drawer mass must be physical; estimate and document it",
    )
    for name in ("red", "blue"):
        box(name + "_geom", bid(name), [0, 0, 0], [0.02] * 3)
        require(
            abs(m.body_mass[bid(name)] - 0.064) < 1e-6
            and m.body_gravcomp[bid(name)] == 0,
            "Physical cube mass/gravity changed",
        )
    dq = m.jnt_dofadr[jid("drawer_slide")]
    require(
        0 <= m.dof_damping[dq] <= 20 and 0 <= m.dof_frictionloss[dq] <= 10,
        "Finite physical calibration bounds for estimated drawer friction/damping",
    )
    for i in range(m.neq):
        require(
            m.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
            and {int(m.eq_obj1id[i]), int(m.eq_obj2id[i])} == {7, 8},
            "Only native gripper mimic equality allowed",
        )
    allowed_bodies = {
        "world",
        "link0",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "hand",
        "left_finger",
        "right_finger",
        "drawer",
        "red",
        "blue",
    }
    require(
        {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}
        == allowed_bodies,
        "No extra physical helper bodies",
    )
    return dict(
        interface=True,
        physical_structure=True,
        native_control=True,
        no_hidden_data=True,
    )

