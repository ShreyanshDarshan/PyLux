# python/taichi/examples/simulation/fractal.py

import taichi as ti

ti.init(arch=ti.gpu)

n = 500
fac = 1
frame_fac = 1.0 #1000.0
history = ti.field(dtype=float, shape=(n, n))
history.fill(-1)  # Initialize all cells to dead
tmp_history = ti.field(dtype=float, shape=history.shape)
display_img = ti.field(dtype=float, shape=history.shape)

cursor_pos = ti.field(dtype=ti.types.vector(2, int), shape=(n*2))

@ti.func
def ti_max(a: float, b: float) -> float:
    return a if a > b else b

@ti.kernel
def update(x: int, y: int, frame: float):
    print (f"Updating cell at ({x}, {y}) with frame {frame}")
    for i, j in history:
        selected_frame = -1
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = (i + di) % n, (j + dj) % n
                # nbr_offset = ti.Vector([di, dj])
                nbr_frame_id = history[ni, nj]
                if nbr_frame_id < 0:
                    continue
                nbr_frame_id = int(nbr_frame_id)
                charge_pos = cursor_pos[int(nbr_frame_id)]
                rvec_center = ti.Vector([i, j]) - charge_pos
                dist_kernel_center = rvec_center.norm()
                dist_circle_center = (frame - nbr_frame_id)
                if (dist_kernel_center - dist_circle_center) <= 1:
                    if selected_frame < nbr_frame_id:
                        selected_frame = nbr_frame_id
                # print (f"(i,j)({i}, {j}) - offset({di}, {dj}) -> nbr({ni}, {nj}) | nbr_frame_id: {nbr_frame_id} | charge_pos: {charge_pos} | selected_frame: {selected_frame} | dist_kernel_center: {dist_kernel_center} | dist_circle_center: {dist_circle_center}")

        tmp_history[i, j] = selected_frame
    for i, j in history:
        history[i, j] = tmp_history[i, j]
    history[x, y] = frame

# Add interactivity to toggle cells

gui = ti.GUI("Lux", res=(n*fac, n*fac))

@ti.kernel
def make_visible():
    for i, j in history:
        display_img[i, j] = history[i, j] / 1000.0

frame = 0
while gui.running:
    # breakpoint()
    make_visible()
    gui.set_image(display_img)
    gui.show()
    
    # for e in gui.get_events(ti.GUI.PRESS):
    #     if e.key == ti.GUI.ESCAPE:
    #         gui.running = False
    #     elif e.key == ti.GUI.SPACE:
    #         pause_state = not pause_state
    #         print(f"Pause state: {pause_state}")
    # if gui.is_pressed(ti.GUI.LMB):
    #     x, y = gui.get_cursor_pos()
    #     toggle_cell(int(x * n), int(y * n))
    #     print(f"Toggled cell at ({int(x * n)}, {int(y * n)})")    

    mx, my = gui.get_cursor_pos()
    if mx <= 0 or mx >= 1 or my <= 0 or my >= 1:
        mx = 0.5
        my = 0.5
    mx = int(mx * n)
    my = int(my * n)

    cursor_pos[int(frame)] = ti.Vector([mx, my])
    update(mx, my, frame)
    # breakpoint()
    frame += 1