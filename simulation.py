import taichi as ti
import math
from charge import Charge
from typing import Tuple, List


class World:
    def __init__(
        self, world_size: Tuple[int, int] = (500, 500), charges: List[Charge] = []
    ):
        self.world_size = world_size
        self.charges = charges


@ti.data_oriented
class Simulator:
    def __init__(self, world: World, grid_size: Tuple[int, int] = (500, 500)):
        self.world = world
        self.grid_size = grid_size
        self.step = 0

    def update(self):
        for charge in self.world.charges:
            charge.update_pos_history(self.step)
            charge.update_hind_field(self.step)

        self.world.charges[0].position = ti.Vector(
            [
                500 + 100 * math.sin(self.step * 0.01),
                250 + 100 * math.cos(self.step * 0.01),
            ]
        )

        self.step += 1
