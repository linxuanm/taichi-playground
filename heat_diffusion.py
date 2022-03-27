import taichi as ti


ti.init(debug=True)

n = 64
scatter = 8
dt = 1e-3
dx = 1
diffusivity = 250
temp_range = (0, 300)

pixels = ti.Vector.field(3, ti.f32, shape=(n*scatter, n*scatter))
T_n = ti.field(ti.f32, shape=(n, n))
T_n1 = ti.field(ti.f32, shape=(n, n))

# copied to color heat
@ti.func
def get_color(v, vmin, vmax):
    c = ti.Vector([1.0, 1.0, 1.0])

    if v < vmin:
        v = vmin
    if v > vmax:
        v = vmax
    dv = vmax - vmin

    if v < (vmin + 0.25 * dv):
        c[0] = 0
        c[1] = 4 * (v-vmin) / dv
    elif v < (vmin + 0.5 * dv):
        c[0] = 0
        c[2] = 1 + 4 * (vmin + 0.25*dv -v) / dv
    elif v < (vmin + 0.75*dv):
        c[0] = 4 * (v - vmin -0.5 * dv) / dv
        c[2] = 0
    else:
        c[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv
        c[2] = 0

    return c


@ti.kernel
def diffuse():
    c = dt * diffusivity / dx**2
    for i, j in T_n:
        T_n1[i, j] = T_n[i, j]
        if i - 1 >= 0:
            T_n1[i, j] += c * (T_n[i - 1, j] - T_n[i, j])
        if i + 1 < n:
            T_n1[i, j] += c * (T_n[i + 1, j] - T_n[i, j])
        if j - 1 >= 0:
            T_n1[i, j] += c * (T_n[i, j - 1] - T_n[i, j])
        if j + 1 < n:
            T_n1[i, j] += c * (T_n[i, j + 1] - T_n[i, j])


@ti.kernel
def print_color():
    for i, j in ti.ndrange(n, n):
        for dx, dy in ti.ndrange(scatter, scatter):
            pixels[i*scatter+dx, j*scatter+dy] = get_color(T_n1[i, j], *temp_range)


@ti.kernel
def add_heat(pos_x: ti.f32, pos_y: ti.f32):
    for i, j in T_n:
        dist = ti.Vector([pos_x - i, pos_y - j]).norm()
        if dist < 5:
            T_n[i, j] += 15


gui = ti.GUI('Heat Diffusion', res=(n * scatter, n * scatter))

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()

    if gui.is_pressed(ti.GUI.LMB):
        x, y = gui.get_cursor_pos()
        add_heat(x * n, y * n)

    diffuse()
    T_n.copy_from(T_n1)
    print_color()

    gui.set_image(pixels)
    gui.show()
