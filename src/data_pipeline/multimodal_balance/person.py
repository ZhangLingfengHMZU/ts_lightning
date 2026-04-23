"""Template Person class for multimodal balance data synthesis."""

from __future__ import annotations

from typing import Tuple


class Person:
    """Domain template describing one subject and generation interfaces."""

    def __init__(
        self,
        subject_id: str,
        age: int,
        gender: str,
        height_cm: float,
        weight_kg: float,
        foot_length_cm: float,
        stance_width_cm: float,
        health_state: int = 0,
    ) -> None: 
        self.subject_id = subject_id
        self.age = age
        self.gender = gender
        self.height_cm = height_cm
        self.weight_kg = weight_kg
        self.foot_length_cm = foot_length_cm
        self.stance_width_cm = stance_width_cm
        self.health_state = health_state

    def _get_error_profile(self) -> Tuple[bool, bool, bool]: 

        if self.health_state == 0:
            self.is_cog_error = False
            self.is_emg_error = False
            self.is_cop_error = False
            return self.is_cog_error, self.is_emg_error, self.is_cop_error

    def generate_rest_EO(self, duration: float): ...

    def generate_semg_data(self, task_type: str): ...

    def generate_cop_data(self, task_type: str): ...
