from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .geometry import CandidateViewpoint


class BaseUAVPolicy:
    name = "base"

    def select(self, observation: Dict[str, Any], candidates: Sequence[CandidateViewpoint]) -> int:
        raise NotImplementedError


class FixedUAVPolicy(BaseUAVPolicy):
    name = "fixed"

    def __init__(self, index: int = 0):
        self.index = index

    def select(self, observation: Dict[str, Any], candidates: Sequence[CandidateViewpoint]) -> int:
        return max(0, min(self.index, len(candidates) - 1))


class RandomUAVPolicy(BaseUAVPolicy):
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def select(self, observation: Dict[str, Any], candidates: Sequence[CandidateViewpoint]) -> int:
        return self.rng.randrange(len(candidates))


class EgoFollowPolicy(BaseUAVPolicy):
    name = "ego_follow"

    def select(self, observation: Dict[str, Any], candidates: Sequence[CandidateViewpoint]) -> int:
        ego = observation["ego"]
        ego_yaw = float(ego["pose"]["yaw"])
        best_idx = 0
        best_score = -1e18
        for idx, cand in enumerate(candidates):
            off = cand.local_offset
            score = off.x - abs(off.y) * 0.25 + off.z * 0.15
            score += math.cos(math.radians(ego_yaw)) * off.x * 0.05
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx


class IntersectionCenterPolicy(BaseUAVPolicy):
    name = "intersection_center"

    def select(self, observation: Dict[str, Any], candidates: Sequence[CandidateViewpoint]) -> int:
        ego = observation["ego"]
        junction = ego.get("junction")
        best_idx = 0
        best_score = -1e18
        for idx, cand in enumerate(candidates):
            off = cand.local_offset
            score = cand.weight + off.z * 0.2 - math.sqrt(off.x * off.x + off.y * off.y) * 0.02
            if junction:
                if abs(off.x) < 5.0 and abs(off.y) < 5.0:
                    score += 2.0
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx


class OcclusionHeuristicPolicy(BaseUAVPolicy):
    name = "occlusion_heuristic"

    def select(self, observation: Dict[str, Any], candidates: Sequence[CandidateViewpoint]) -> int:
        ego = observation["ego"]
        ego_loc = ego["pose"]["position"]
        vehicles = observation.get("vehicles", [])
        best_idx = 0
        best_score = -1e18
        for idx, cand in enumerate(candidates):
            pos = cand.local_offset
            # V0 heuristic: prefer higher, more centered, and richer traffic density.
            score = cand.weight + pos.z * 0.3 - (abs(pos.x) + abs(pos.y)) * 0.03
            visible_proxy = 0.0
            for vehicle in vehicles:
                cur = vehicle["pose"]["position"]
                dx = cur["x"] - ego_loc["x"]
                dy = cur["y"] - ego_loc["y"]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 60.0:
                    visible_proxy += max(0.0, 60.0 - dist) / 60.0
            score += visible_proxy * 0.1
            if ego.get("junction"):
                score += 0.5
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx


def build_policy(name: str, index: int = 0, seed: Optional[int] = None) -> BaseUAVPolicy:
    normalized = name.lower()
    if normalized == "fixed":
        return FixedUAVPolicy(index=index)
    if normalized == "random":
        return RandomUAVPolicy(seed=seed)
    if normalized in {"follow", "ego_follow", "ego-follow"}:
        return EgoFollowPolicy()
    if normalized in {"intersection", "intersection_center", "center"}:
        return IntersectionCenterPolicy()
    if normalized in {"heuristic", "occlusion", "active"}:
        return OcclusionHeuristicPolicy()
    raise ValueError(f"Unknown policy: {name}")

