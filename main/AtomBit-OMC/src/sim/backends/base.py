from abc import ABC, abstractmethod
from typing import Any

from ase.calculators.calculator import Calculator


class BaseCalculatorBackend(ABC):
    def __init__(self, backend_name: str, backend_config: dict[str, Any]):
        self.backend_name = backend_name
        self.backend_config = backend_config

    @abstractmethod
    def get_calculator(self) -> Calculator:
        raise NotImplementedError

    @abstractmethod
    def model_info(self) -> dict[str, Any]:
        raise NotImplementedError
