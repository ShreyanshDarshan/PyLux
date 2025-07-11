# python/taichi/examples/simulation/fractal.py

import taichi as ti
import math
from simulation import World, Simulator
from visualization import Visualizer
from charge import Charge

ti.init(arch=ti.gpu)


world = World(
    world_size=(500, 500, 500),
    charges=[
        Charge(
            ti.Vector([250, 250, 250]), ti.Vector([0, 0, 0]), grid_size=(500, 500, 500)
        )
    ],
)
sim = Simulator(world=world, grid_size=(500, 500, 300))
viz = Visualizer(world=sim.world)

while viz.running:
    viz.render()

    # mx, my = gui.get_cursor_pos()
    # if mx <= 0 or mx >= 1 or my <= 0 or my >= 1:
    #     mx = 0.5
    #     my = 0.5
    # mx = int(mx * sim.grid_size[0])
    # my = int(my * sim.grid_size[1])

    sim.update()
