__all__ = [
    "VisionPolicy",
    "SimpleLaneVisionPolicy",
    "COMMAND_TO_INDEX",
    "TcpLiteModel",
    "command_to_index",
]


def __getattr__(name):
    if name == "VisionPolicy":
        from .base import VisionPolicy

        return VisionPolicy
    if name == "SimpleLaneVisionPolicy":
        from .simple_lane import SimpleLaneVisionPolicy

        return SimpleLaneVisionPolicy
    if name in {"COMMAND_TO_INDEX", "TcpLiteModel", "command_to_index"}:
        from . import tcp_lite

        return getattr(tcp_lite, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
