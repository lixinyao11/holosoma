"""Import the same URDFs as Isaac; never substitute the generic G1 MJCF."""
from pathlib import Path
import xml.etree.ElementTree as ET
import hashlib
import json
import mujoco


def source_path(path):
    if path.startswith("@holosoma/"):
        import holosoma
        return Path(holosoma.__file__).parent / path[len("@holosoma/"):]
    return Path(path)


def load_urdf_spec(path, robot_cfg, audit_dir):
    path = Path(path).resolve()
    tree = ET.parse(path)
    root = tree.getroot()
    ext = root.find("mujoco")
    if ext is None:
        ext = ET.SubElement(root, "mujoco")
    compiler = ext.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(ext, "compiler")
    compiler.set("fusestatic", "false")
    compiler.set("discardvisual", "false")
    compiler.set("balanceinertia", "false")
    manifest = {"source":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"meshes":{}}
    for mesh in root.findall(".//mesh"):
        raw = mesh.get("filename")
        p = Path(raw)
        if not p.is_absolute():
            if raw.startswith("package://"):
                raw = raw[len("package://"):].split("/",1)[-1]
            p = path.parent / raw
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        mesh.set("filename",str(p))
        manifest["meshes"][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    if not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in spec.joints):
        spec.bodies[1].add_freejoint(name="floating_base")
    # URDF visual and collision geometry are imported without replacing meshes.
    if robot_cfg is not None:
        for name, armature, friction in zip(robot_cfg.dof_names,robot_cfg.dof_armature_list,robot_cfg.dof_joint_friction_list):
            joint = spec.joint(name)
            joint.armature = armature
            joint.frictionloss = friction
            joint.damping = (0., 0., 0.)
    converted_capsules = []
    if robot_cfg is not None and robot_cfg.asset.replace_cylinder_with_capsule:
        for geom in spec.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER and (geom.contype or geom.conaffinity):
                # Verified against the frozen Isaac USD: radius and cylinder length
                # remain unchanged; MuJoCo size[1] is that length / 2.
                geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
                converted_capsules.append({"name":geom.name,"radius":float(geom.size[0]),
                    "height":float(2*geom.size[1])})
    manifest["cylinder_to_capsule"] = converted_capsules
    model = spec.compile()
    manifest["compiled"] = {"nbody":model.nbody,"njnt":model.njnt,"ngeom":model.ngeom,
        "mass":model.body_mass.tolist(),"inertia":model.body_inertia.tolist(),
        "body_names":[model.body(i).name for i in range(model.nbody)]}
    manifest["conversion"] = "original URDF; absolute mesh paths; retained fixed bodies; floating base; configured joint armature"
    (Path(audit_dir)/("robot_asset.json" if robot_cfg is not None else "object_asset.json")).write_text(json.dumps(manifest,indent=2))
    return spec
