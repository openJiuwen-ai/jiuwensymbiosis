"""New action bindings must participate in adapter capability validation."""

import pytest

from jiuwensymbiosis.api.actions import (
    APPROACH_TARGET,
    LOCATE_TARGET,
    NAVIGATE_TO_ENTITY,
    NAVIGATE_TO_POSE,
    SIM_GET_ENTITY_STATE,
    SIM_GET_ROBOT_STATE,
    SIM_LIST_ENTITIES,
    implements,
)
from scripts.validate_adapter import _check_capability_actions


@pytest.mark.parametrize(
    "spec",
    [
        LOCATE_TARGET,
        APPROACH_TARGET,
        NAVIGATE_TO_POSE,
        NAVIGATE_TO_ENTITY,
        SIM_LIST_ENTITIES,
        SIM_GET_ROBOT_STATE,
        SIM_GET_ENTITY_STATE,
    ],
    ids=lambda spec: spec.name,
)
def test_adapter_implementing_only_new_action_satisfies_its_capability(spec):
    class Api:
        @implements(spec)
        def action(self, **kwargs):
            return {}

    assert _check_capability_actions(Api) == []


def test_simulation_truth_requires_an_implemented_action():
    class Api:
        capability = "simulation.truth"

    warnings = _check_capability_actions(Api)
    assert len(warnings) == 1
    assert warnings[0][0] == "simulation.truth"
