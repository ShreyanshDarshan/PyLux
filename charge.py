import taichi as ti
import math


@ti.data_oriented
class Charge:
    def __init__(
        self, position: ti.Vector, velocity: ti.Vector, grid_size=(50, 50, 50)
    ):
        self.position = position
        self.velocity = velocity
        self.acceleration = ti.Vector([0.0, 0.0, 0.0])
        self.charge = 1.0

        self.last_position = ti.Vector([0.0, 0.0, 0.0])
        self.last_velocity = ti.Vector([0.0, 0.0, 0.0])

        self.hind_field = ti.field(dtype=int, shape=grid_size)
        self.hind_field.fill(-1)
        self.hind_buffer = ti.field(dtype=int, shape=self.hind_field.shape)

        pos_history_size = int(
            math.sqrt(grid_size[0] ** 2 + grid_size[1] ** 2 + grid_size[2] ** 2) * 1.2
        )
        self.pos_history = ti.Vector.field(3, dtype=float, shape=(pos_history_size,))
        self.pos_buffer = ti.Vector.field(3, dtype=float, shape=(pos_history_size,))

        self.acc_history = ti.Vector.field(3, dtype=float, shape=(pos_history_size,))
        self.acc_buffer = ti.Vector.field(3, dtype=float, shape=(pos_history_size,))

        self.electric_field = ti.Vector.field(3, dtype=float, shape=grid_size)
        self.electric_field.fill(ti.Vector([0.0, 0.0, 0.0]))

        self.energy_field = ti.field(dtype=float, shape=grid_size)
        self.energy_field.fill(0.0)

    @ti.kernel
    def _update_hind_field(self, step: int):
        grid_size = self.hind_field.shape

        for i, j, k in self.hind_field:
            selected_hind = -1
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    for dk in range(-1, 2):
                        ni, nj, nk = (
                            (i + di) % grid_size[0],
                            (j + dj) % grid_size[1],
                            (k + dk) % grid_size[2],
                        )
                        nbr_hind = int(self.hind_field[ni, nj, nk])
                        if nbr_hind < 0:
                            continue
                        historical_pos = self.pos_history[step - nbr_hind]
                        rvec_center = ti.Vector([i, j, k]) - historical_pos
                        dist_kernel_center = rvec_center.norm()
                        radius_circle = step - nbr_hind
                        if (dist_kernel_center - radius_circle) <= 1:
                            if selected_hind < nbr_hind:
                                selected_hind = nbr_hind

            self.hind_buffer[i, j, k] = selected_hind

        for i, j, k in self.hind_field:
            self.hind_field[i, j, k] = self.hind_buffer[i, j, k]
        x, y, z = (
            int(self.pos_history[0][0]),
            int(self.pos_history[0][1]),
            int(self.pos_history[0][2]),
        )
        self.hind_field[x, y, z] = step

    @ti.kernel
    def update_electric_field(self, step: int):
        for i, j, k in self.electric_field:
            hind = self.hind_field[i, j, k]
            if hind >= 0:
                historical_pos = self.pos_history[step - hind]
                rvec_center = ti.Vector([i, j, k]) - historical_pos
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
                self.electric_field[i, j, k] = electric_strength

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
        if step == 1:
            self.last_velocity = self.velocity

        self.acceleration = self.velocity - self.last_velocity

        # print(
        #     f"Step: {step}, Last Position: {self.last_position}, Last Velocity: {self.last_velocity}"
        # )
        # print(
        #     f"Step: {step}, Position: {self.position}, Velocity: {self.velocity}, Acceleration: {self.acceleration}"
        # )

        self._update_histories()
        self.pos_history[0] = self.position
        self.acc_history[0] = self.acceleration

        self.last_position = self.position
        self.last_velocity = self.velocity

    def update_hind_field(self, step: int):
        self.update_histories(step)
        self._update_hind_field(step)

    @ti.kernel
    def calculate_energy(self):
        for i, j, k in self.energy_field:
            e_sq = self.electric_field[i, j, k].norm_sqr()
            if ti.math.isnan(e_sq):
                e_sq = 0.0
            self.energy_field[i, j, k] = e_sq
