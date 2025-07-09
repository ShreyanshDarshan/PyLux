# python/taichi/examples/simulation/fractal.py

import taichi as ti
import math

ti.init(arch=ti.gpu)

@ti.data_oriented
class Simulation:
    def __init__(self, grid_size=(500, 500)):
        self.grid_size = grid_size

        # Index field that points to position history of particle
        self.hind_field = ti.field(dtype=float, shape=grid_size)
        self.hind_field.fill(-1)  # Initialize all cells to dead
        self.hind_buffer = ti.field(dtype=float, shape=self.hind_field.shape)
        self.display_img = ti.field(dtype=float, shape=self.hind_field.shape)
        pos_history_size = int(math.sqrt(grid_size[0]**2 + grid_size[1]**2) * 1.2)
        self.pos_history = ti.Vector.field(2, dtype=float, shape=(pos_history_size,))

    @ti.func
    def max(self, a: float, b: float) -> float:
        return a if a > b else b

    @ti.kernel
    def update(self, x: int, y: int, frame: float):
        for i, j in self.hind_field:
            selected_frame = -1
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = (i + di) % self.grid_size[0], (j + dj) % self.grid_size[1]
                    nbr_frame_id = int(self.hind_field[ni, nj])
                    if nbr_frame_id < 0:
                        continue
                    charge_pos = self.pos_history[nbr_frame_id]
                    rvec_center = ti.Vector([i, j]) - charge_pos
                    dist_kernel_center = rvec_center.norm()
                    radius_circle = (frame - nbr_frame_id)
                    if (dist_kernel_center - radius_circle) <= 1:
                        if selected_frame < nbr_frame_id:
                            selected_frame = nbr_frame_id
                    # print (f"(i,j)({i}, {j}) - offset({di}, {dj}) -> nbr({ni}, {nj}) | nbr_frame_id: {nbr_frame_id} | charge_pos: {charge_pos} | selected_frame: {selected_frame} | dist_kernel_center: {dist_kernel_center} | dist_circle_center: {dist_circle_center}")

            self.hind_buffer[i, j] = selected_frame
        for i, j in self.hind_field:
            self.hind_field[i, j] = self.hind_buffer[i, j]
        self.hind_field[x, y] = frame

    def update_pos_history(self, frame: int, x: int, y: int):
        self.pos_history[frame] = ti.Vector([x, y])

    @ti.kernel
    def make_visible(self):
        for i, j in self.hind_field:
            self.display_img[i, j] = self.hind_field[i, j] / 1000.0

gui = ti.GUI("Lux", res=(1000, 500))

sim = Simulation(grid_size=(1000, 500))
frame = 0
while gui.running:
    # breakpoint()
    sim.make_visible()
    gui.set_image(sim.display_img)
    gui.show() 

    mx, my = gui.get_cursor_pos()
    if mx <= 0 or mx >= 1 or my <= 0 or my >= 1:
        mx = 0.5
        my = 0.5
    mx = int(mx * sim.grid_size[0])
    my = int(my * sim.grid_size[1])

    sim.update_pos_history(frame, mx, my)
    sim.update(mx, my, frame)

    frame += 1