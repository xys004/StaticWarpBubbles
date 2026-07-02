"""
pywarpbubblefactory/ansatze/warp_shell.py
=========================================
WarpShell ansatz wrapping pyWarpFactory's native implementation.
"""

from typing import Dict, Any
from .base import Ansatz
from warpfactory.generator.warp_shell import create_warp_shell_metric
from warpfactory.generator.base import Metric

class WarpShellAnsatz(Ansatz):
    def __init__(self, grid_size=(1, 50, 50, 50), grid_scale=(1, 0.1, 0.1, 0.1), world_center=(0, 0, 0, 0)):
        super().__init__("WarpShell", grid_size, grid_scale, world_center)

    def build(self, params: Dict[str, Any]) -> Metric:
        """
        Required params:
        - mass: Total mass of the shell
        - r_inner: Inner radius
        - r_outer: Outer radius
        - r_buff: Buffer distance
        - sigma: Sigmoid sharpness
        - smooth_factor: Integer smoothing iterations
        - v_warp: Warp velocity
        """
        return create_warp_shell_metric(
            grid_size=self.grid_size,
            grid_scale=self.grid_scale,
            world_center=self.world_center,
            mass=params.get('mass', 1.0),
            r_inner=params.get('r_inner', 1.0),
            r_outer=params.get('r_outer', 2.0),
            r_buff=params.get('r_buff', 0.5),
            sigma=params.get('sigma', 2.0),
            smooth_factor=int(params.get('smooth_factor', 2)),
            v_warp=params.get('v_warp', 0.0),
            do_warp=params.get('v_warp', 0.0) != 0.0
        )
