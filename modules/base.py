from abc import ABC, abstractmethod
from typing import Any

class BaseModule(ABC):
    def __init__(self, config: dict = {}):
        self.config = config

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__