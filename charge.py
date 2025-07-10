import taichi as ti
import math


@ti.data_oriented
class Charge:
    def __init__(self, position: ti.Vector, velocity: ti.Vector, grid_size=(500, 500)):
        self.position = position
        self.velocity = velocity

        # Index field that points to position history of particle
        self.hind_field = ti.field(dtype=int, shape=grid_size)
        self.hind_field.fill(-1)  # Initialize all cells to dead
        self.hind_buffer = ti.field(dtype=int, shape=self.hind_field.shape)

        pos_history_size = int(math.sqrt(grid_size[0] ** 2 + grid_size[1] ** 2) * 1.2)
        self.pos_history = ti.Vector.field(2, dtype=float, shape=(pos_history_size,))

        # Initialize electric field
        # self.electric_field = ti.Vector.field(2, dtype=float, shape=grid_size)
        # self.electric_field.fill(ti.Vector([0.0, 0.0]))

    @ti.kernel
    def _update_hind_field(self, step: int):
        grid_size = self.hind_field.shape

        for i, j in self.hind_field:
            selected_hind = -1
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = (i + di) % grid_size[0], (j + dj) % grid_size[1]
                    nbr_hind = int(self.hind_field[ni, nj])
                    if nbr_hind < 0:
                        continue
                    charge_pos = self.pos_history[nbr_hind]
                    rvec_center = ti.Vector([i, j]) - charge_pos
                    dist_kernel_center = rvec_center.norm()
                    radius_circle = step - nbr_hind
                    if (dist_kernel_center - radius_circle) <= 1:
                        if selected_hind < nbr_hind:
                            selected_hind = nbr_hind
                    # print (f"(i,j)({i}, {j}) - offset({di}, {dj}) -> nbr({ni}, {nj}) | nbr_frame_id: {nbr_frame_id} | charge_pos: {charge_pos} | selected_frame: {selected_frame} | dist_kernel_center: {dist_kernel_center} | dist_circle_center: {dist_circle_center}")

            self.hind_buffer[i, j] = selected_hind

        # Update the hind field with the buffer
        for i, j in self.hind_field:
            self.hind_field[i, j] = self.hind_buffer[i, j]
        x, y = int(self.pos_history[int(step)])
        self.hind_field[x, y] = step

    def update_pos_history(self, step: int):
        self.pos_history[step] = self.position

    def update_hind_field(self, step: int):
        self.update_pos_history(step)
        self._update_hind_field(step)
