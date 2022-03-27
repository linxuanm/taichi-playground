import taichi as ti


ti.init(debug=True)

dt = 1e-5
planets_limit = 20

num_planets = ti.field(ti.i32, shape=())

x = ti.Vector.field(2, ti.f32, shape=planets_limit, needs_grad=True)
v = ti.Vector.field(2, ti.f32, shape=planets_limit)
potential = ti.field(ti.f32, shape=(), needs_grad=True)


@ti.kernel
def compute_potential(n: ti.i32):
    for i, j in ti.ndrange(n, n):
        dist = (x[i] - x[j]).norm(1e-3)
        potential[None] += -2 / dist


@ti.kernel
def update():
    for i in range(num_planets[None]):
        v[i] += dt * -x.grad[i]

    for i in range(num_planets[None]):
        x[i] += dt * v[i]


def step():
    with ti.Tape(potential):
        compute_potential(num_planets[None])

    update()


def make_planet(pos_x: ti.i32, pos_y: ti.i32):
    new_id = num_planets[None]
    num_planets[None] += 1

    x[new_id] = [pos_x, pos_y]


gui = ti.GUI('Gravity', res=(720, 720), background_color=0x000000)

while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == ti.GUI.LMB and num_planets[None] < planets_limit:
            make_planet(e.pos[0], e.pos[1])

    for i in range(50):
        step()
    gui.circles(x.to_numpy(), radius=10)
    gui.show()
