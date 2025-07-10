import taichi as ti
from simulation import Simulator, World
from charge import Charge


@ti.data_oriented
class Visualizer:
    def __init__(self, world: World, win_name="Lux"):
        self.world = world
        self.display_img = ti.field(dtype=float, shape=world.world_size)

        # Initialize the display image
        self.display_img.fill(0.0)

        self.gui = ti.GUI("Lux", res=world.world_size)

    @ti.kernel
    def make_visible(self):
        for i, j in self.display_img:
            self.display_img[i, j] = self.world.charges[0].hind_field[i, j] / 1000.0

    def render(self):
        self.gui.set_image(self.display_img)
        self.gui.show()

    @property
    def running(self):
        return self.gui.running
