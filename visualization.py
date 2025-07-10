import taichi as ti
from simulation import Simulator, World
from charge import Charge


@ti.data_oriented
class Visualizer:
    def __init__(self, world: World, win_name="Lux"):
        self.world = world
        self.display_img = ti.Vector.field(3, dtype=float, shape=world.world_size)

        # Initialize the display image with zeros
        self.display_img.fill(ti.Vector([0.0, 0.0, 0.0]))

        self.gui = ti.GUI("Lux", res=world.world_size)

    @ti.kernel
    def show_charge_hind(self):
        for i, j in self.display_img:
            color_bw = self.world.charges[0].hind_field[i, j] / 1000.0
            color = ti.Vector([color_bw, color_bw, color_bw])
            self.display_img[i, j] = color

    @ti.kernel
    def show_charge_electric_field(self):
        for i, j in self.display_img:
            color_rg = abs(self.world.charges[0].electric_field[i, j]) * 10000.0
            color_rgb = ti.Vector([color_rg[0], color_rg[1], 0.0])
            self.display_img[i, j] = color_rgb

        self.display_img[500, 250] = ti.Vector(
            [1.0, 1.0, 1.0]
        )  # Highlight the center position

    def render(self):
        self.show_charge_electric_field()
        self.gui.set_image(self.display_img)
        self.gui.show()

    @property
    def running(self):
        return self.gui.running
