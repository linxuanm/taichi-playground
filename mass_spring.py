import taichi as ti


ti.init(debug=True)

particles_limit = 256
mass = 1
ground_y = 0.05
conn_range = 0.15
G = ti.Vector([0, -9.81])
dt = 1e-3

num_particles = ti.var(dt=ti.i32, shape=())
stiffness = ti.var(dt=ti.f32, shape=())
damping = ti.var(dt=ti.f32, shape=())

x = ti.Vector(2, dt=ti.f32, shape=particles_limit)
v = ti.Vector(2, dt=ti.f32, shape=particles_limit)

A = ti.Matrix(2, 2, dt=ti.f32, shape=(particles_limit, particles_limit))
b = ti.Vector(2, dt=ti.f32, shape=particles_limit)

rest_length = ti.var(ti.f32, shape=(particles_limit, particles_limit))
fixed = ti.var(ti.i32, shape=particles_limit)

stiffness[None] = 10000
damping[None] = 0


@ti.kernel
def update():
    n = num_particles[None]
    for i in range(n):

        v[i] *= ti.exp(-dt * damping)
        force = G * mass

        for j in range(n):
            if rest_length[i, j] == 0:
                continue

            ij = x[j] - x[i]
            magnitude = stiffness[None] * (ij.norm() - rest_length[i, j])
            force += magnitude * ij.normalized()

        v[i] += force / mass * dt

    for i in range(n):
        if x[i][1] < ground_y:
            x[i][1] = ground_y
            v[i][1] = 0

    for i in range(n):
        if not fixed[i]:
            x[i] += v[i] * dt


@ti.kernel
def make_node(pos_x: ti.f32, pos_y: ti.f32, is_fixed: ti.i32):
    curr = num_particles[None]
    x[curr] = [pos_x, pos_y]
    v[curr] = [0, 0]
    fixed[curr] = is_fixed
    num_particles[None] += 1

    for i in range(curr):
        if (x[curr] - x[i]).norm() <= conn_range:
            rest_length[i, curr] = 0.1
            rest_length[curr, i] = 0.1


gui = ti.GUI('Mass Spring', res=(720, 720), background_color=0x000000)

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == ti.GUI.LMB and num_particles[None] < particles_limit:
            make_node(e.pos[0], e.pos[1], gui.is_pressed('Shift'))

    for _ in range(10):
        update()

    nodes = x.to_numpy()[: num_particles[None]]
    fix_status = fixed.to_numpy()[: num_particles[None]]

    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if rest_length[i, j] == 0:
                continue

            gui.line(begin=nodes[i], end=nodes[j], radius=2, color=0xAAAAAA)

    non_fixed_nodes = nodes[fix_status == 0]
    fixed_nodes = nodes[fix_status == 1]
    gui.circles(non_fixed_nodes, color=0x5BC2E2, radius=5)
    gui.circles(fixed_nodes, color=0xFF7B5B, radius=5)

    gui.line(
        begin=(0.0, ground_y),
        end=(1.0, ground_y),
        color=0xFFFFFF,
        radius=1
    )

    gui.show()
