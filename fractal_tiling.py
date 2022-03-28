import taichi as ti


ti.init(debug=True)

layers = 6
shape = (1280, 720)

pixels = ti.Vector.field(4, ti.f32, shape=shape)


@ti.func
def fract(vec):
    return vec - ti.floor(vec)


@ti.func
def smoothstep(edge1, edge2, v):
    assert(edge1 != edge2)
    t = (v-edge1) / float(edge2-edge1)
    t = clamp(t, 0.0, 1.0)

    return (3-2 * t) * t**2


@ti.func
def clamp(v, v_min, v_max):
    return ti.min(ti.max(v, v_min), v_max)


@ti.kernel
def render(t: ti.f32):
    for i, j in pixels:
        color = ti.Vector([0.0, 0.0, 0.0, 0.0])
        pos = ti.Vector([256 * i / shape[1], 256 * j / shape[1]]) + t
        for k in ti.ndrange(layers):
            floor = ti.floor(pos)
            fract = fract(pos)

            grid = ti.sin(floor.x * 7.0 + 31.0 * floor.y + 0.01 * t)
            const = ti.Vector([0.035, 0.01, 0.0, 0.7]) * 13.545317
            w = grid + const
            final = w - ti.floor(w)

            intensity = final * 2 * smoothstep(0.45, 0.55, final[3])
            color += intensity * ti.sqrt(
                16.0 * fract.x * fract.y * (1.0 - fract.x) * (1.0 - fract.y)
            )

            pos /= 2
            color /= 2

        pixels[i, j] = color


gui = ti.GUI('Fractal Tiling', res=shape)
t = 0

while True:
    render(t)
    t += 0.25

    gui.set_image(pixels)
    gui.show()
