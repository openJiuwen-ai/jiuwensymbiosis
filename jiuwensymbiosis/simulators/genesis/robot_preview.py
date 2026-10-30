"""Fixed-base URDF inspection, explicitly separate from validated robot control."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class RobotPreview:
    """Keep the imported articulated model at a reproducible inspection posture."""

    def __init__(
        self, scene, path: Path, *, name: str, render_urdf: Path | None = None, file_meshes_are_zup: bool = True
    ):
        import genesis as gs

        self.path = path
        self.name = name
        self.render_urdf = render_urdf
        self.target: np.ndarray | None = None
        self.bounds: list[list[float]] | None = None
        self.xml = ET.parse(path).getroot()
        visual_xml = ET.parse(self.render_urdf).getroot() if self.render_urdf else self.xml
        self.visible_links = {
            link.get("name") for link in visual_xml.findall("link") if link.find("visual") is not None
        }
        self.entity = scene.add_entity(
            morph=gs.morphs.URDF(
                file=str(path),
                fixed=True,
                merge_fixed_links=False,
                collision=False,
                decimate=False,
                convexify=False,
                file_meshes_are_zup=file_meshes_are_zup,
            ),
        )

    def build(self) -> None:
        import numpy as np

        self.target = np.zeros(self.entity.n_dofs)
        positions: dict[str, float] = {}
        pending = list(self.xml.findall("joint"))
        while pending:
            progress = False
            for joint in pending[:]:
                name = joint.attrib["name"]
                mimic = joint.find("mimic")
                if mimic is not None and mimic.get("joint") not in positions:
                    continue
                value = (
                    0.0
                    if mimic is None
                    else (
                        float(mimic.get("multiplier", "1")) * positions[mimic.attrib["joint"]]
                        + float(mimic.get("offset", "0"))
                    )
                )
                positions[name] = value
                if joint.get("type") != "fixed":
                    indices = self.entity.get_joint(name).dofs_idx_local
                    if len(indices) == 1:
                        self.target[indices[0]] = value
                pending.remove(joint)
                progress = True
            if not progress:
                raise ValueError("Mimic joint references contain a cycle")
        self.hold()
        box = self.visual_bounds()
        self.entity.set_pos((0, 0, -float(box[0, 2]) + 0.005))
        self.bounds = self.visual_bounds().tolist()

    def visual_bounds(self):
        import torch

        # Collision guides in the original export must neither render nor enlarge
        # the framing/ground-placement bounds of the material-restored robot.
        vertices = torch.cat([v.get_vverts()[0] for v in self.entity.vgeoms if v.link.name in self.visible_links])
        return torch.stack((vertices.amin(dim=0), vertices.amax(dim=0))).cpu().numpy()

    def hold(self) -> None:
        if self.target is None:
            raise RuntimeError("Build the robot inspection posture before holding it")
        self.entity.set_dofs_position(self.target, zero_velocity=True)

    def snapshot(self) -> dict:
        joints = [j for j in self.entity.joints if j.n_dofs]
        return {
            "name": self.name,
            "source_format": "URDF",
            "urdf": str(self.path),
            "links": self.entity.n_links,
            "dofs": self.entity.n_dofs,
            "joint_names": [j.name for j in joints],
            "joint_positions": self.entity.get_dofs_position()[0].cpu().tolist(),
            "fixed_base": True,
            "control_mode": "kinematic_inspection",
            "collision": False,
            "planning_connected": False,
            "appearance": "source_usd_pbr" if self.render_urdf else "urdf_basic",
        }
