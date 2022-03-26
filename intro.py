import taichi as ti


ti.init(debug=True)

n = 360
pixels = ti.var(dt=ti.f32, shape=(n * 2, n))


@ti.func
def complex_sqr(z):
    return ti.Vector([z[0] ** 2 - z[1] ** 2, 2 * z[0] * z[1]])


@ti.kernel
def update(t: ti.f32):
    for i, j in pixels:
        const = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2

        curr = 0
        while z.norm() < 20 and curr < 50:
            z = complex_sqr(z) + const
            curr += 1

        pixels[i, j] = curr * 0.02

i = 0
gui = ti.GUI("Julia Set", res=(n * 2, n))

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()

    update(i)
    gui.set_image(pixels)
    gui.show()
    i += 0.03
