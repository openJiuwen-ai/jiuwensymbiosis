# Prepare and audit robot collision models

`robot_model.prepare_robot_model()` resolves URDF resources and writes a derived model manifest.
`collision_audit.audit_loaded_model()` reads a built Genesis entity before controller initialization.
Neither operation steps physics or commands motion. The generic modules contain no robot names,
default exclusions, or controller gains. New robots supply their own policy and lifecycle integration.

## Installation and supported assets

Install `.[simulator-collision]` for CPU geometry preparation. It does not include Genesis;
load auditing requires Genesis 1.4.4. A GUI installation can combine `.[gui,genesis,simulator-collision]`.
The decomposition cache uses POSIX locking; the validated platform is Linux x86_64.

Supported inputs are single-root URDF trees with fixed, revolute, continuous or prismatic joints,
acyclic mimic references, and box/sphere/cylinder or OBJ/STL/PLY triangle-mesh collisions.
Multiple collisions per link are supported. Mesh scales must be positive; units are metres, radians and kilograms.
Origins, scales, inertial XML and joints are retained without guessing units. Unsupported geometry, negative scales,
closed loops and unresolved resources fail explicitly. Dynamic USD robots, deformables and multiple environments
are outside this contract. Visual textures and indirect dependencies are not covered by collision integrity checks.

## Preparation and explicit policies

Run from the repository root, avoiding core framework initialization:

```bash
python jiuwensymbiosis/simulators/genesis/robot_model.py /absolute/robot.urdf \
  --output outputs/robot-audit/example --package robot_description=/absolute/robot_description
```

Relative and absolute mesh paths are supported; package URIs require explicit mappings.
`model.json` records source/derived URDF and mesh SHA-256 hashes, structure, transforms, policy and tool versions.
Source assets are immutable and hashes are checked again before and after loading.
The asset export manifest pairs source physics and visual assets; the derived manifest identifies the actual collision model.

No policy is selected implicitly. Use `--policy /absolute/policy.yaml` with `schema_version: 1`, a `name`,
`decomposed_links`, `disabled_links` mapping each name to a reason, and optional CoACD `parameters`.
Invalid input or decomposition fails without a single-hull fallback. CoACD's threshold is a search target,
not a guaranteed surface-error bound. See the [Chinese guide](../../zh/how-to/audit-robot-models.md) for a policy example.

`configs/simulators/collision/moz1_*_collision.yaml` contains opt-in examples for specific MOZ1 assets:
three decomposed hand links and two excluded loop links. Verify the source USD collision flags before reuse.
There is no GUI policy selector. Ordinary mobile mode passes no policy and preserves source collisions.
`Moz1Base` remains a MOZ1 controller; its names and gains are not a generic robot interface.

The output's `collision_parts/` cache is keyed by source bytes, type, parameters, tool versions and preparation-code hash.
Repeated calls to the same output directory reuse it; separate instance directories do not share caches.
Every hit verifies hashes, convexity, closure, winding, volume and bounds. Process locks and atomic publication
prevent partial caches. Part basenames include the full key to avoid importer asset-name collisions.

## Loading and evidence lifecycle

`--load-audit` builds a CPU scene without stepping. Defaults are fixed base, no decimation, self-collision off.
Use `--floating --decimate` for the mobile runtime's base/simplification settings; `--self-collision` enables self-collision.
Shared `admission_morph_options()` preserves authored frames (align=False), disables fixed-link merging and implicit decomposition,
enables collision/convexify,
and sets watertighten=5. The report records effective import settings, solver flags and SDK versions.
The CLI destroys both the scene and Genesis on success or failure.

For an existing caller-owned scene:

```python
import genesis as gs
from jiuwensymbiosis.simulators.genesis.robot_model import prepare_robot_model, admission_morph_options
from jiuwensymbiosis.simulators.genesis.collision_audit import audit_loaded_model

model = prepare_robot_model(source, output)  # source/output are pathlib.Path instances
entity = scene.add_entity(gs.morphs.URDF(file=model["urdf"], **admission_morph_options(fixed=False, decimate=True)))
scene.build()
report = audit_loaded_model(entity, model, output / "loaded-model.json")
# Initialize this robot's controller only after admission; the caller owns cleanup.
```

`RuntimeScene → Moz1Base.build()` audits before motor initialization. Empty collision models are rejected.
`info.model_admission` exposes status/report location. `robot_model_instance_id` identifies construction and survives reset.
Reports live in the state directory's `robot-models/<robot_model_instance_id>/`, outside temporary rendering files.
Scene shutdown writes a `closed` marker. Success and failure evidence is retained without automatic deletion;
operators should budget storage and archive/remove only closed directories after deciding evidence is no longer needed.
Direct `Moz1Base` callers must provide a persistent model directory. Kinematic `RobotPreview` is outside admission.

Reports include geometry/source counts, vertex/face fingerprints, bounds, 26-direction support matching, masks and allowed pairs.
Analytic sphere/cylinder comparisons allow 2.5% radius tessellation error; reprocessed comparisons uniformly allow 10 micrometres
and use `processed_index`, not an assumed URDF collision index. Reprocessing tests loading consistency, not SDK algorithm accuracy.
Unavailable readback is `unverifiable` with the exception cause retained; inconsistent geometry is `error`. Both refuse admission.
A `pass` does not establish exact concave surfaces, stable contact, stopping, reset or grasp success. Those require physical acceptance.
