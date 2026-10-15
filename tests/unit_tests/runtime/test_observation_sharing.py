from unittest.mock import Mock

from jiuwensymbiosis.agent.trace import TraceRail
from jiuwensymbiosis.rails.visual_feedback import VisualFeedbackRail
from jiuwensymbiosis.runtime.events import TaskEventRail
from tests.helpers import FakeCtx, FakeInputs, make_mock_session


async def test_trace_feedback_and_task_events_share_each_post_action_frame(tmp_path, monkeypatch):
    session = make_mock_session()
    capture = Mock(wraps=session.env.get_observation)
    monkeypatch.setattr(session.env, "get_observation", capture)
    trace = TraceRail(session, workspace=str(tmp_path), save_frames=True)
    feedback = VisualFeedbackRail(session)
    events = TaskEventRail(Mock(), session)
    await trace.before_invoke(FakeCtx(FakeInputs(conversation_id="sharing", query="test")))
    capture.reset_mock()
    ctx = FakeCtx(FakeInputs(tool_name="goto_xyzr", tool_args={}, tool_result={"ok": True}))
    try:
        for _ in range(2):
            for rail in (trace, feedback, events):
                await rail.before_tool_call(ctx)
            for rail in (trace, feedback, events):
                await rail.after_tool_call(ctx)
        assert capture.call_count == 2
    finally:
        trace.close()
