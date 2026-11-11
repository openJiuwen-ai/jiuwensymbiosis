# MOZ1 collision policy examples

These policies are opt-in CLI/code examples, not runtime defaults. Each decomposes three hand links
and disables two loop links. Revalidate against your asset; similar names do not establish equivalent geometry.

Source: `SpiritAIAssets/Robots/moz1/moz1_omni_gripper_double_narrow.usda`.
Verified authored `physics:collisionEnabled=false` on the following geometry prims (2026-11-10):

- `/MOZ1/Geometry/base_link/leg01/leg02/leg03/waist01/waist02/waist03/left01/left02/left03/left04/left05/left06/left07/left_gripper_base_link/left_hand_narrow_loop_Link/narrow_loop_Link_1/narrow_loop_Link`
- `/MOZ1/Geometry/base_link/leg01/leg02/leg03/waist01/waist02/waist03/left01/left02/left03/left04/left05/left06/left07/left_gripper_base_link/left_hand_wide_loop_Link/wide_loop_Link_1/wide_loop_Link`
- `/MOZ1/Geometry/base_link/leg01/leg02/leg03/waist01/waist02/waist03/right01/right02/right03/right04/right05/right06/right07/right_gripper_base_link/right_hand_narrow_loop_Link/narrow_loop_Link_1/narrow_loop_Link`
- `/MOZ1/Geometry/base_link/leg01/leg02/leg03/waist01/waist02/waist03/right01/right02/right03/right04/right05/right06/right07/right_gripper_base_link/right_hand_wide_loop_Link/wide_loop_Link_1/wide_loop_Link`

The composed source depends on these layers (relative to `Robots/moz1/`); all hashes are SHA-256.
For a local asset, open it with `pxr.Usd.Stage.Open`, resolve each prim with `GetPrimAtPath`,
and require both `HasAuthoredValueOpinion()` and `Get() is False` on `physics:collisionEnabled`.

| Layer | SHA-256 |
| --- | --- |
| `moz1_omni_gripper_double_narrow.usda` | `9de2f62ab15cd587cc40de50a8669fca9fee8b4b952f6a1c8b3d2b1fb8b0ec0b` |
| `SubUSDs/base.usda` | `334044988d78b73b0a09964574886b6b0debee38fb6a4b4798f6057e9af1992a` |
| `moz1_omni_gripper_full.usda` | `8532fdcdeb649403b05bcd19fe741f36ca5e0df35205c1caf959a7d62728373d` |
| `SubUSDs/physx.usda` | `a73e2f5af9bd028657a4901288628cd862bbc520dbe9a7915cba23af5f7ccae4` |
| `SubUSDs/robot.usda` | `4e1017a36cc21c3f166d7573abce87d434e704a14b1b4e8418c4afd155148bf2` |
| `SubUSDs/instances.usda` | `74e7688ad1ac145904dc386be409b827fc4bf63c0c0d2b5b02f18dbbf739947b` |
| `SubUSDs/physics.usda` | `8153406e5555a253d3fb760fb1b2dd9ef4e278b1b855b033a6c024acdd9f5924` |
| `SubUSDs/OmniWheel_10.usda` | `ef816f945a62ddb9b847d36b9ed994068047def74f7e677cde0d17cb5fc13b96` |
| `SubUSDs/OmniWheel_10_physics.usda` | `2fe61a08a2ca24ab6a2d8a8fbf18db3534ec467e56dbf8f05ba9a5aa60db6dc9` |
| `SubUSDs/OmniWheel_10_sensor.usda` | `b19f49ec4e5e0ccbff58de8fbd2fc9d40497f1a99186bf8676b7f403e299bb45` |
| `SubUSDs/OmniWheel_10_base.usdc` | `687c3a37ec78e19109237af52daee6539775abdf625aa2904bfdb3320a452100` |
| `SubUSDs/OmniWheel_10_robot.usda` | `e6be18a33ec1e309a7c37061a28f44ec61592fff765de40ccb2d869f686287a7` |
| `SubUSDs/double_narrow_geometry.usda` | `1270c04ed56a55928d68665e1dfe6b02d7f890efbc85eb83b617b16a1279596a` |
| `SubUSDs/materials.usda` | `0c1fcb08ac0e81a08ee4edbb69107a33d4b25b25ed1a2fe8d081d000aa5e2259` |
| `SubUSDs/geometries.usd` | `3c474db7cf6a9db5e24d541983246eff799c55a650cfdbdd3ade7c49bf552fff` |
