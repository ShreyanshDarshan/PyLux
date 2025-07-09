# python/taichi/examples/simulation/fractal.py

import taichi as ti

ti.init(arch=ti.gpu)

n = 700
pixels = ti.field(dtype=float, shape=(n, n))
pixels.fill(0)  # Initialize all cells to dead
new_pixels = ti.field(dtype=float, shape=pixels.shape)
new_pixels.fill(0)  # Initialize new pixels

cursor_pos = []

@ti.func
def ti_max(a: float, b: float) -> float:
    return a if a > b else b

@ti.kernel
def update(x: int, y: int, frame: float):
    for i, j in pixels:
        max_val = pixels[i, j]
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = (i + di) % n, (j + dj) % n
                max_val = ti_max(max_val, pixels[ni, nj])

        new_pixels[i, j] = max_val
    for i, j in pixels:
        pixels[i, j] = new_pixels[i, j]
    pixels[x, y] = frame

# Add interactivity to toggle cells

gui = ti.GUI("Lux", res=(n, n))

pause_state = False

while gui.running:
    # breakpoint()
    gui.set_image(pixels)  # Convert bool to u8 for display
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

    cursor_pos.append((mx, my))
    if not pause_state:
        update(mx, my, gui.frame / 1000.0)