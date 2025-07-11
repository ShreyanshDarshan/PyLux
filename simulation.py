import taichi as ti
import math
from charge import Charge
from typing import Tuple, List
import numpy as np


class World:
    def __init__(
        self,
        world_size: Tuple[int, int, int] = (500, 500, 500),
        charges: List[Charge] = [],
    ):
        self.world_size = world_size
        self.charges = charges


@ti.data_oriented
class Simulator:
    def __init__(self, world: World, grid_size: Tuple[int, int, int] = (500, 500, 500)):
        self.world = world
        self.grid_size = grid_size
        self.step = 0

    def update(self):
        if self.step <= 10:
            self.world.charges[0].position = ti.Vector(
                [
                    251 + 1 * math.sin(self.step * math.pi / 10 - math.pi / 2),
                    250,  # + 50 * math.cos(self.step * 0.01),
                    250,
                ]
            )
        for charge in self.world.charges:
            charge.update_hind_field(self.step)
            charge.update_electric_field(self.step)

        self.world.charges[0].calculate_energy()
        energy = self.world.charges[0].energy_field.to_torch().cuda().sum()
        print(f"Step {self.step}, Energy: {energy.item()}")
        # breakpoint()

        self.step += 1
