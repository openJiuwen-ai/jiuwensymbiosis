"""Auditable USD-to-URDF compatibility fixes; never edit source asset layers."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


def iter_link_geometry(prim):
    """Prune nested rigid bodies, rather than exporting their meshes on every ancestor."""
    from pxr import Usd, UsdGeom, UsdPhysics

    iterator = iter(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
    for child in iterator:
        if child == prim:
            continue
        if child.HasAPI(UsdPhysics.RigidBodyAPI) or child.IsA(UsdPhysics.Joint):
            iterator.PruneChildren()
        elif child.IsA(UsdGeom.Gprim):
            yield child


def prepare_export_stage(stage, *, name_override_attribute: str | None = None) -> dict:
    """Fix the session layer; exporters may explicitly supply a custom name attribute."""
    from pxr import UsdPhysics

    stage.SetEditTarget(stage.GetSessionLayer())
    # Instance proxies cannot carry overrides. Deinstance only the export view.
    while True:
        instances = [p for p in stage.Traverse() if p.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    disabled = []
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is False:
            disabled.append(str(prim.GetPath()))
            prim.SetActive(False)
    names = {}
    for is_joint in (False, True):
        if is_joint:
            prims = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
        else:
            prims = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        originals = []
        for prim in prims:
            name = prim.GetMetadata("displayName")
            if not name and name_override_attribute:
                name = prim.GetAttribute(name_override_attribute).Get()
            originals.append(name or prim.GetName())
        counts = Counter(originals)
        for prim, name in zip(prims, originals, strict=True):
            unique = str(prim.GetPath()).strip("/").replace("/", "__") if counts[name] > 1 else name
            prim.SetMetadata("displayName", unique)
            names[str(prim.GetPath())] = unique
    joints = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        name = names[str(prim.GetPath())]
        angular = prim.IsA(UsdPhysics.RevoluteJoint)
        joint = {"path": str(prim.GetPath()), "angular": angular}
        for key, attr in {
            "effort": "drive:angular:physics:maxForce" if angular else "drive:linear:physics:maxForce",
            "velocity": "physxJoint:maxJointVelocity",
        }.items():
            value = prim.GetAttribute(attr).Get()
            if value is not None:
                joint[key] = float(value) * (math.pi / 180 if key == "velocity" and angular else 1)
        # usd-core can read authored PhysX metadata without the PhysX plugin.
        api_schemas = prim.GetMetadata("apiSchemas")
        schemas = api_schemas.GetAppliedItems() if api_schemas is not None else prim.GetAppliedSchemas()
        for schema in schemas:
            if str(schema).startswith("PhysxMimicJointAPI:"):
                instance = str(schema).split(":", 1)[1]
                prefix = f"physxMimicJoint:{instance}:"
                targets = prim.GetRelationship(prefix + "referenceJoint").GetTargets()
                reference = stage.GetPrimAtPath(targets[0]) if targets else None
                if not reference or reference.IsA(UsdPhysics.RevoluteJoint) != angular:
                    raise ValueError("Only same-unit one-axis mimic joints are supported by this export profile")
                gearing = prim.GetAttribute(prefix + "gearing").Get()
                offset = prim.GetAttribute(prefix + "offset").Get()
                reference_name = names.get(str(targets[0]))
                if reference_name is None:
                    raise ValueError("Mimic reference joint is missing from the export view")
                # PhysX: q + gearing*q_ref + offset = 0; angular offsets are degrees.
                joint["mimic"] = {
                    "joint": reference_name,
                    "multiplier": -float(gearing if gearing is not None else 1),
                    "offset": -float(offset or 0) * (math.pi / 180 if angular else 1),
                }
        joints[name] = joint
    return {"disabled_joints": disabled, "prim_names": names, "joints": joints}


def finish_export(path: Path, metadata: dict) -> dict:
    """Repair exported mimic coefficients/limits and remove the synthetic free-base wrapper."""
    tree = ET.parse(path)
    robot = tree.getroot()
    changes: list[dict[str, Any]] = []
    for joint in list(robot.findall("joint")):
        name = joint.attrib["name"]
        parent = joint.find("parent")
        if parent is None:
            raise ValueError(f"Missing parent for joint {name}")
        if name == "floating_base" and joint.get("type") == "floating" and parent.get("link") == "world":
            world = next((link for link in robot.findall("link") if link.get("name") == "world"), None)
            if world is None or list(world):
                raise ValueError("Unexpected nonempty synthetic world link")
            robot.remove(joint)
            robot.remove(world)
            changes.append({"joint": name, "change": "base fixed/free state is selected by Genesis morph"})
            continue
        source = metadata["joints"].get(name, {})
        if "mimic" in source:
            mimic = joint.find("mimic")
            if mimic is None:
                mimic = ET.SubElement(joint, "mimic")
            before = dict(mimic.attrib)
            mimic.attrib.update({k: str(v) for k, v in source["mimic"].items()})
            changes.append({"joint": name, "mimic_before": before, "mimic_after": dict(mimic.attrib)})
        limit = joint.find("limit")
        if limit is not None:
            for field in ("effort", "velocity"):
                value = source.get(field)
                valid = value is not None and value > 0 and math.isfinite(value)
                if valid and float(limit.get(field, "0")) == 0:
                    limit.set(field, str(value))
                    changes.append({"joint": name, "field": field, "value": value, "source": "USD drive/PhysX"})
            if joint.get("type") == "revolute" and limit.get("lower") == "-inf" and limit.get("upper") == "inf":
                joint.set("type", "continuous")
                del limit.attrib["lower"]
                del limit.attrib["upper"]
            for field in ("effort", "velocity"):
                if field not in limit.attrib:
                    limit.set(field, "0")
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return {"changes": changes, **validate_urdf(path)}


def validate_urdf(path: Path) -> dict:
    """Reject ambiguous names, cycles, missing meshes and dangling mimic references."""
    robot = ET.parse(path).getroot()
    link_names = [entry.attrib["name"] for entry in robot.findall("link")]
    joint_names = [entry.attrib["name"] for entry in robot.findall("joint")]
    for kind, names in (("link", link_names), ("joint", joint_names)):
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate {kind} names")
    parents = {}
    for joint in robot.findall("joint"):
        child_element, parent_element = joint.find("child"), joint.find("parent")
        if child_element is None or parent_element is None:
            raise ValueError(f"Invalid parent/child for {joint.get('name')}")
        child, parent = child_element.get("link"), parent_element.get("link")
        if child not in link_names or parent not in link_names or child in parents:
            raise ValueError(f"Invalid parent/child for {joint.get('name')}")
        parents[child] = parent
        mimic = joint.find("mimic")
        if mimic is not None and mimic.get("joint") not in joint_names:
            raise ValueError("Unknown mimic joint")
    roots = set(link_names) - parents.keys()
    if len(roots) != 1 or len(link_names) != len(joint_names) + 1:
        raise ValueError("Robot must have one root and a tree structure")
    for name in link_names:
        seen = set()
        while True:
            parent = parents.get(name)
            if parent is None:
                break
            if name in seen:
                raise ValueError("Cycle in URDF tree")
            seen.add(name)
            name = parent
    for mesh in robot.findall(".//mesh"):
        if not (path.parent / mesh.attrib["filename"]).is_file():
            raise ValueError(f"Missing mesh: {mesh.attrib['filename']}")
    return {"root": next(iter(roots)), "links": len(link_names), "joints": len(joint_names)}
