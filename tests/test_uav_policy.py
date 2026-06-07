from __future__ import annotations

from carlaair_active_world.geometry import CandidateViewpoint, Vector3
from carlaair_active_world.policies import OcclusionHeuristicPolicy


def test_occlusion_heuristic_prefers_forward_lead_view_over_top_hover():
    policy = OcclusionHeuristicPolicy()
    candidates = [
        CandidateViewpoint("front_lead_high", Vector3(28.0, 0.0, 32.0), 1.4),
        CandidateViewpoint("top", Vector3(0.0, 0.0, 26.0), 0.8),
    ]
    observation = {
        "ego": {"pose": {"position": {"x": 0.0, "y": 0.0}, "yaw": 0.0}, "junction": False},
        "vehicles": [],
    }

    assert policy.select(observation, candidates) == 0
