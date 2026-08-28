import numpy as np

from openpi.serving.exp_websocket_policy_server import (
    ExpWebsocketPolicyServer,
    ServerRTCState,
    split_rtc_payload,
)


def test_split_rtc_payload_preserves_legacy_payload():
    payload = {"state": np.zeros(2), "images": {"cam": np.zeros((2, 2, 3))}}

    observation, rtc_context = split_rtc_payload(payload)

    assert rtc_context is None
    assert observation is payload


def test_split_rtc_payload_extracts_optional_context():
    payload = {
        "state": np.zeros(2),
        "images": {"cam": np.zeros((2, 2, 3))},
        "rtc_context": {"executed_action_id": 2},
    }

    observation, rtc_context = split_rtc_payload(payload)

    assert observation["state"].shape == (2,)
    assert rtc_context == {"executed_action_id": 2}
    assert "rtc_context" not in observation


def test_server_rtc_state_builds_reanchoring_context_from_executed_action_id():
    state = ServerRTCState()

    assert state.build_model_context(5) == {"prev_action_chunk": None, "inference_delay": 0}

    raw_actions = np.arange(12, dtype=np.float32).reshape(4, 3)
    reference_state = np.array([1.0, 2.0], dtype=np.float32)
    state.record_chunk(raw_actions, reference_state=reference_state, base_action_id=5)

    context = state.build_model_context(7)
    assert np.array_equal(context["prev_action_chunk"], raw_actions[2:])
    assert np.array_equal(context["prev_action_reference_state"], reference_state)
    assert context["inference_delay"] == 2


def test_disabled_rtc_strips_client_execution_progress_before_normal_inference():
    class FakePolicy:
        def __init__(self):
            self.calls = []

        def infer(self, observation, rtc_context=None):
            self.calls.append((observation, rtc_context))
            return {"actions": np.ones((1, 2), dtype=np.float32)}

    policy = FakePolicy()
    server = ExpWebsocketPolicyServer(policy, enable_rtc=False)
    observation = {"state": np.zeros(2, dtype=np.float32)}

    action = server._infer(observation, {"executed_action_id": 3}, ServerRTCState())

    assert "actions" in action
    assert policy.calls == [(observation, None)]


def test_enabled_rtc_passes_server_owned_context_and_retains_raw_actions():
    class FakePolicy:
        def __init__(self):
            self.calls = []

        def infer(self, observation, rtc_context=None):
            self.calls.append((observation, rtc_context))
            return {
                "actions": np.ones((1, 2), dtype=np.float32),
                "raw_model_actions": np.arange(6, dtype=np.float32).reshape(2, 3),
            }

    policy = FakePolicy()
    server = ExpWebsocketPolicyServer(policy, enable_rtc=True)
    observation = {"state": np.array([1.0, 2.0], dtype=np.float32)}
    rtc_state = ServerRTCState()

    action = server._infer(observation, {"executed_action_id": 3}, rtc_state)

    assert policy.calls == [
        (
            observation,
            {"prev_action_chunk": None, "inference_delay": 0},
        )
    ]
    assert "raw_model_actions" not in action
    assert np.array_equal(rtc_state.raw_actions, np.arange(6, dtype=np.float32).reshape(2, 3))
    assert np.array_equal(rtc_state.reference_state, observation["state"])
    assert rtc_state.chunk_base_action_id == 3
