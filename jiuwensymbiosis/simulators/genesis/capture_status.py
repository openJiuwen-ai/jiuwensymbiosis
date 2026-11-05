"""Refresh measured state on the sole physics owner during frozen RGB-D."""


def refresh_capture_status(runtime, mailbox, *, measure_base, now):
    state = mailbox.read_status()
    if state.get("state") != "ready" or state.get("scene", {}).get("instance_id") != runtime.info["instance_id"]:
        return False
    # Every refreshed dynamic field is read now; no timestamp-only heartbeat.
    observed = measure_base(runtime.robot)
    robot = runtime.robot.snapshot()
    contacts = runtime.indoor.contacts(runtime.robot) if runtime.indoor is not None else None
    objects = [body.get_pos()[0].cpu().tolist() for body in runtime.bodies]
    state.update(
        base_observation=observed,
        robot=robot,
        scene_contacts=contacts,
        object_positions=objects,
        base_motion=runtime.motion.result,
        steps=runtime.steps,
        sim_time=runtime.steps * runtime.settings.dt,
        measured_at=now(),
    )
    mailbox.write_status(state)
    return True
