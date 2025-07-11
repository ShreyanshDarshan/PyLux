import taichi as ti
import math


@ti.data_oriented
class Charge:
    def __init__(self, position: ti.Vector, velocity: ti.Vector, grid_size=(500, 500)):
        self.position = position
        self.velocity = velocity
        self.acceleration = ti.Vector([0.0, 0.0])
        self.charge = 1.0

        self.last_position = ti.Vector([0.0, 0.0])
        self.last_velocity = ti.Vector([0.0, 0.0])

        # Index field that points to history-index to use at each point in space
        self.hind_field = ti.field(dtype=int, shape=grid_size)
        self.hind_field.fill(-1)  # Initialize all cells to dead
        self.hind_buffer = ti.field(dtype=int, shape=self.hind_field.shape)

        pos_history_size = int(math.sqrt(grid_size[0] ** 2 + grid_size[1] ** 2) * 1.2)
        self.pos_history = ti.Vector.field(2, dtype=float, shape=(pos_history_size,))
        self.pos_buffer = ti.Vector.field(2, dtype=float, shape=(pos_history_size,))

        self.acc_history = ti.Vector.field(2, dtype=float, shape=(pos_history_size,))
        self.acc_buffer = ti.Vector.field(2, dtype=float, shape=(pos_history_size,))

        # Initialize electric field
        self.electric_field = ti.Vector.field(2, dtype=float, shape=grid_size)
        self.electric_field.fill(ti.Vector([0.0, 0.0]))

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
                    historical_pos = self.pos_history[step - nbr_hind]
                    rvec_center = ti.Vector([i, j]) - historical_pos
                    dist_kernel_center = rvec_center.norm()
                    radius_circle = step - nbr_hind
                    if (dist_kernel_center - radius_circle) <= 1:
                        if selected_hind < nbr_hind:
                            selected_hind = nbr_hind

            self.hind_buffer[i, j] = selected_hind

        # Update the hind field with the buffer
        for i, j in self.hind_field:
            self.hind_field[i, j] = self.hind_buffer[i, j]
        x, y = int(self.pos_history[0])
        self.hind_field[x, y] = step

    @ti.kernel
    def update_electric_field(self, step: int):
        # grid_size = self.electric_field.shape
        for i, j in self.electric_field:
            hind = self.hind_field[i, j]
            if hind >= 0:
                historical_pos = self.pos_history[step - hind]
                rvec_center = ti.Vector([i, j]) - historical_pos
                historical_acc = self.acc_history[step - hind]
                dist_kernel_center = rvec_center.norm()
                hist_acc_perp = (
                    historical_acc
                    - historical_acc.dot(rvec_center.normalized())
                    * rvec_center.normalized()
                )

                electric_strength = (
                    self.charge * hist_acc_perp / (dist_kernel_center + 1e-6)
                )
                self.electric_field[i, j] = electric_strength

    @ti.kernel
    def _update_histories(self):
        for i in self.pos_buffer:
            if i > 0:
                self.pos_buffer[i] = self.pos_history[i - 1]

        for i in self.pos_history:
            self.pos_history[i] = self.pos_buffer[i]

        for i in self.acc_buffer:
            if i > 0:
                self.acc_buffer[i] = self.acc_history[i - 1]
        for i in self.acc_history:
            self.acc_history[i] = self.acc_buffer[i]

    def update_histories(self, step: int):
        if step == 0:
            self.last_position = self.position
            self.last_velocity = self.velocity

        self.velocity = self.position - self.last_position
        self.acceleration = self.velocity - self.last_velocity

        self._update_histories()
        self.pos_history[0] = self.position
        self.acc_history[0] = self.acceleration
        # breakpoint()

        self.last_position = self.position
        self.last_velocity = self.velocity

    def update_hind_field(self, step: int):
        self.update_histories(step)
        self._update_hind_field(step)
