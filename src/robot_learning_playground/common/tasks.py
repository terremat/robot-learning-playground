from dataclasses import dataclass


@dataclass(frozen=True)
class PickPlaceTask:
    object_name: str
    target_name: str