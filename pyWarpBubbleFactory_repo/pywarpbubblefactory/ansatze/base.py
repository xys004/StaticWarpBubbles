"""
pywarpbubblefactory/ansatze/base.py
===================================
Base class for parametric metric ansatze.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

# We'll use pyWarpFactory's Metric class as the standardized output
from warpfactory.generator.base import Metric

class Ansatz(ABC):
    """
    Abstract base class for a parametric metric family.
    """
    def __init__(self, name: str, grid_size: tuple, grid_scale: tuple, world_center: tuple):
        self.name = name
        self.grid_size = grid_size
        self.grid_scale = grid_scale
        self.world_center = world_center

    @abstractmethod
    def build(self, params: Dict[str, Any]) -> Metric:
        """
        Builds and returns the Metric object given a set of parameters.
        """
        pass
